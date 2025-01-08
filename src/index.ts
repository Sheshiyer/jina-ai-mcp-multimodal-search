#!/usr/bin/env node
import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import {
  CallToolRequestSchema,
  ErrorCode,
  ListToolsRequestSchema,
  McpError,
} from '@modelcontextprotocol/sdk/types.js';
import axios from 'axios';
import dotenv from 'dotenv';

dotenv.config();

const JINA_API_KEY = process.env.JINA_API_KEY;
if (!JINA_API_KEY) {
  throw new Error('JINA_API_KEY environment variable is required');
}

interface SearchArgs {
  query: string;
  collection: string;
  limit?: number;
}

interface ImageSearchArgs {
  imageUrl: string;
  collection: string;
  limit?: number;
}

interface CrossModalSearchArgs {
  query: string;
  mode: 'text2image' | 'image2text';
  collection: string;
  limit?: number;
}

interface JinaSearchResponse {
  results: Array<{
    id: string;
    score: number;
    data: Record<string, any>;
  }>;
}

interface JinaErrorResponse {
  message: string;
  code: number;
}

class JinaServer {
  private server: Server;
  private axiosInstance;

  constructor() {
    this.server = new Server(
      {
        name: 'jina-ai-server',
        version: '0.1.0',
      },
      {
        capabilities: {
          tools: {},
        },
      }
    );

    this.axiosInstance = axios.create({
      baseURL: 'https://api.jina.ai/v1',
      headers: {
        'Authorization': `Bearer ${JINA_API_KEY}`,
        'Content-Type': 'application/json',
      },
    });

    this.setupToolHandlers();
    
    this.server.onerror = (error) => console.error('[MCP Error]', error);
    process.on('SIGINT', async () => {
      await this.server.close();
      process.exit(0);
    });
  }

  private setupToolHandlers() {
    this.server.setRequestHandler(ListToolsRequestSchema, async () => ({
      tools: [
        {
          name: 'semantic_search',
          description: 'Perform semantic/neural search on text documents',
          inputSchema: {
            type: 'object',
            properties: {
              query: {
                type: 'string',
                description: 'Search query text',
              },
              collection: {
                type: 'string',
                description: 'Collection name to search in',
              },
              limit: {
                type: 'number',
                description: 'Maximum number of results',
                default: 10,
              },
            },
            required: ['query', 'collection'],
          },
        },
        {
          name: 'image_search',
          description: 'Search for similar images using an image URL',
          inputSchema: {
            type: 'object',
            properties: {
              imageUrl: {
                type: 'string',
                description: 'URL of the query image',
              },
              collection: {
                type: 'string',
                description: 'Collection name to search in',
              },
              limit: {
                type: 'number',
                description: 'Maximum number of results',
                default: 10,
              },
            },
            required: ['imageUrl', 'collection'],
          },
        },
        {
          name: 'cross_modal_search',
          description: 'Perform text-to-image or image-to-text search',
          inputSchema: {
            type: 'object',
            properties: {
              query: {
                type: 'string',
                description: 'Text query or image URL',
              },
              mode: {
                type: 'string',
                enum: ['text2image', 'image2text'],
                description: 'Search mode',
              },
              collection: {
                type: 'string',
                description: 'Collection name to search in',
              },
              limit: {
                type: 'number',
                description: 'Maximum number of results',
                default: 10,
              },
            },
            required: ['query', 'mode', 'collection'],
          },
        },
      ],
    }));

    this.server.setRequestHandler(CallToolRequestSchema, async (request) => {
      try {
        const args = request.params.arguments || {};
        
        switch (request.params.name) {
          case 'semantic_search': {
            const searchArgs = this.validateSearchArgs(args);
            return await this.handleSemanticSearch(searchArgs);
          }
          case 'image_search': {
            const imageArgs = this.validateImageSearchArgs(args);
            return await this.handleImageSearch(imageArgs);
          }
          case 'cross_modal_search': {
            const crossModalArgs = this.validateCrossModalSearchArgs(args);
            return await this.handleCrossModalSearch(crossModalArgs);
          }
          default:
            throw new McpError(
              ErrorCode.MethodNotFound,
              `Unknown tool: ${request.params.name}`
            );
        }
      } catch (error) {
        const err = error as Error & { response?: { data?: JinaErrorResponse } };
        throw new McpError(
          ErrorCode.InternalError,
          `Jina API error: ${err.response?.data?.message || err.message}`
        );
      }
    });
  }

  private validateSearchArgs(args: Record<string, unknown>): SearchArgs {
    if (typeof args.query !== 'string' || typeof args.collection !== 'string') {
      throw new McpError(ErrorCode.InvalidParams, 'Invalid semantic search arguments');
    }
    return {
      query: args.query,
      collection: args.collection,
      limit: typeof args.limit === 'number' ? args.limit : undefined,
    };
  }

  private validateImageSearchArgs(args: Record<string, unknown>): ImageSearchArgs {
    if (typeof args.imageUrl !== 'string' || typeof args.collection !== 'string') {
      throw new McpError(ErrorCode.InvalidParams, 'Invalid image search arguments');
    }
    return {
      imageUrl: args.imageUrl,
      collection: args.collection,
      limit: typeof args.limit === 'number' ? args.limit : undefined,
    };
  }

  private validateCrossModalSearchArgs(args: Record<string, unknown>): CrossModalSearchArgs {
    if (
      typeof args.query !== 'string' ||
      typeof args.collection !== 'string' ||
      (args.mode !== 'text2image' && args.mode !== 'image2text')
    ) {
      throw new McpError(ErrorCode.InvalidParams, 'Invalid cross-modal search arguments');
    }
    return {
      query: args.query,
      mode: args.mode,
      collection: args.collection,
      limit: typeof args.limit === 'number' ? args.limit : undefined,
    };
  }

  private async handleSemanticSearch(args: SearchArgs) {
    const response = await this.axiosInstance.post<JinaSearchResponse>(`/collections/${args.collection}/search`, {
      query: args.query,
      limit: args.limit || 10,
      type: 'text',
    });

    return {
      content: [
        {
          type: 'text',
          text: JSON.stringify(response.data.results, null, 2),
        },
      ],
    };
  }

  private async handleImageSearch(args: ImageSearchArgs) {
    const response = await this.axiosInstance.post<JinaSearchResponse>(`/collections/${args.collection}/search`, {
      query: args.imageUrl,
      limit: args.limit || 10,
      type: 'image',
    });

    return {
      content: [
        {
          type: 'text',
          text: JSON.stringify(response.data.results, null, 2),
        },
      ],
    };
  }

  private async handleCrossModalSearch(args: CrossModalSearchArgs) {
    const response = await this.axiosInstance.post<JinaSearchResponse>(`/collections/${args.collection}/search`, {
      query: args.query,
      limit: args.limit || 10,
      type: args.mode === 'text2image' ? 'text2image' : 'image2text',
    });

    return {
      content: [
        {
          type: 'text',
          text: JSON.stringify(response.data.results, null, 2),
        },
      ],
    };
  }

  async run() {
    const transport = new StdioServerTransport();
    await this.server.connect(transport);
    console.error('Jina AI MCP server running on stdio');
  }
}

const server = new JinaServer();
server.run().catch(console.error);
