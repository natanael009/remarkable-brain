import { readFileSync } from 'node:fs';
import OpenAI from 'openai';
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
  const openai = new OpenAI({ apiKey });

  return {
    messages: {
      async create(args: Record<string, unknown>) {
        const request = args as any;
        const content = request.messages[0].content as any[];

        const image = content.find((item) => item.type === 'image');
        const text = content.find((item) => item.type === 'text');
        const tool = request.tools[0];

        if (!image || !text || !tool) {
          throw new Error('Invalid extraction request: expected image, prompt, and tool schema.');
        }

        const response = await openai.chat.completions.create({
          model: request.model,
          max_tokens: request.max_tokens,
          temperature: 0,
          messages: [
            {
              role: 'user',
              content: [
                {
                  type: 'image_url',
                  image_url: {
                    url: `data:${image.source.media_type};base64,${image.source.data}`,
                  },
                },
                {
                  type: 'text',
                  text: text.text,
                },
              ],
            },
          ],
          tools: [
            {
              type: 'function',
              function: {
                name: tool.name,
                description: tool.description,
                parameters: tool.input_schema,
              },
            },
          ],
          tool_choice: {
            type: 'function',
            function: {
              name: 'record_page',
            },
          },
        });

        const toolCall = response.choices[0]?.message.tool_calls?.find(
          (call: any) =>
            call.type === 'function' &&
            call.function.name === 'record_page',
        );

        if (!toolCall) {
          throw new Error('GPT-4o did not return the required record_page tool call.');
        }

        return {
          content: [
            {
              type: 'tool_use',
              name: 'record_page',
              input: JSON.parse(toolCall.function.arguments),
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
