import { readFileSync } from 'node:fs';
import { randomUUID } from 'node:crypto';
import {
  EXTRACTION_TOOL,
  EXTRACTION_PROMPT,
  PageExtractionSchema,
  normalizeEntityType,
  type PageExtraction,
} from './schema.js';

export interface AnthropicLike {
  messages: {
    create(args: Record<string, unknown>): Promise<{
      content: Array<{ type: string; name?: string; input?: unknown }>;
    }>;
  };
}

export async function createAnthropicClient(apiKey: string): Promise<AnthropicLike> {
  const endpoint = process.env.IAEDU_ENDPOINT;
  const channelId = process.env.IAEDU_CHANNEL_ID;

  if (!endpoint || !channelId) {
    throw new Error(
      'Missing IAEDU_ENDPOINT or IAEDU_CHANNEL_ID environment variable.',
    );
  }

  return {
    messages: {
      async create(args: Record<string, unknown>) {
        const request = args as any;
        const content = request.messages[0].content as any[];

        const image = content.find((item) => item.type === 'image');
        const text = content.find((item) => item.type === 'text');
        const tool = request.tools?.[0];

        if (!image || !text || !tool) {
          throw new Error(
            'Invalid extraction request: expected an image, prompt, and output schema.',
          );
        }

        const prompt = [
          text.text,
          '',
          'Return ONLY one valid JSON object.',
          'Do not use Markdown fences or explanatory text.',
          'It must conform exactly to this JSON Schema:',
          JSON.stringify(tool.input_schema),
        ].join('\n');

        const form = new FormData();
        form.set('channel_id', channelId);
        form.set('thread_id', `rm-brain-${randomUUID()}`);
        form.set('user_info', '{}');
        form.set('message', prompt);
        form.set(
          'image',
          new Blob(
            [Buffer.from(image.source.data, 'base64')],
            { type: image.source.media_type },
          ),
          'remarkable-page.png',
        );

        const response = await fetch(endpoint, {
          method: 'POST',
          headers: {
            'x-api-key': apiKey,
          },
          body: form,
        });

        if (!response.ok) {
          throw new Error(
            `IAedu request failed: ${response.status} ${await response.text()}`,
          );
        }

        const body = await response.text();
        let answer = '';

        for (const line of body.split('\n')) {
          const trimmed = line.trim();

          if (!trimmed) continue;

          try {
            const event = JSON.parse(trimmed);

            if (event.type === 'token' && typeof event.content === 'string') {
              answer += event.content;
            }
          } catch {
            // Ignore non-JSON stream lines.
          }
        }

        const json = answer
          .trim()
          .replace(/^```json\s*/i, '')
          .replace(/^```\s*/i, '')
          .replace(/\s*```$/i, '')
          .trim();

        return {
          content: [
            {
              type: 'tool_use',
              name: 'record_page',
              input: JSON.parse(json),
            },
          ],
        };
      },
    },
  };
}

export async function extractPage(opts: {
  imagePath: string;
  model: string;
  client: AnthropicLike;
}): Promise<PageExtraction> {
  const b64 = readFileSync(opts.imagePath).toString('base64');
  const request = {
    model: opts.model,
    max_tokens: 2048,
    tools: [EXTRACTION_TOOL],
    tool_choice: { type: 'tool', name: 'record_page' },
    messages: [
      {
        role: 'user',
        content: [
          { type: 'image', source: { type: 'base64', media_type: 'image/png', data: b64 } },
          { type: 'text', text: EXTRACTION_PROMPT },
        ],
      },
    ],
  };

  let lastErr: unknown;
  for (let attempt = 0; attempt < 2; attempt++) {
    const res = await opts.client.messages.create(request);
    const toolUse = res.content.find((c) => c.type === 'tool_use' && c.name === 'record_page');
    const parsed = PageExtractionSchema.safeParse(toolUse?.input);
    if (parsed.success) {
      // Canonicalize entity types onto the fixed vocabulary so the same real thing isn't split
      // across synonymous types (Location/Place, Item/Topic) into duplicate entities.
      return {
        ...parsed.data,
        entities: parsed.data.entities.map((e) => ({
          name: e.name.trim(),
          type: normalizeEntityType(e.type),
        })),
      };
    }
    lastErr = parsed.error;
  }
  throw new Error(`extraction returned invalid output: ${String(lastErr)}`);
}
