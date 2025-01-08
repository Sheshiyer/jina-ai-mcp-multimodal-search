# Jina AI MCP Server

A Model Context Protocol (MCP) server that provides seamless integration with Jina AI's neural search capabilities. This server enables semantic search, image search, and cross-modal search functionalities through a simple interface.

## 🚀 Features

- **Semantic Search**: Find semantically similar documents using natural language queries
- **Image Search**: Search for visually similar images using image URLs
- **Cross-Modal Search**: Perform text-to-image or image-to-text searches

## 📋 Prerequisites

- Node.js 16 or higher
- A Jina AI account and API key ([Get one here](https://cloud.jina.ai/))
- MCP-compatible environment (e.g., Cline)

## 🛠️ Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd jina-ai-mcp
```

2. Install dependencies:
```bash
npm install
```

3. Create a `.env` file with your Jina AI API key:
```bash
JINA_API_KEY=your_api_key_here
```

4. Build the server:
```bash
npm run build
```

## ⚙️ Configuration

Add the following configuration to your MCP settings file:

```json
{
  "mcpServers": {
    "jina-ai": {
      "command": "node",
      "args": [
        "/path/to/jina-ai-mcp/build/index.js"
      ],
      "env": {
        "JINA_API_KEY": "your_api_key_here"
      }
    }
  }
}
```

## 🔍 Available Tools

### 1. Semantic Search
Perform semantic/neural search on text documents.

```typescript
use_mcp_tool({
  server_name: "jina-ai",
  tool_name: "semantic_search",
  arguments: {
    query: "search query text",
    collection: "your-collection-name",
    limit: 10 // optional, defaults to 10
  }
})
```

### 2. Image Search
Search for similar images using an image URL.

```typescript
use_mcp_tool({
  server_name: "jina-ai",
  tool_name: "image_search",
  arguments: {
    imageUrl: "https://example.com/image.jpg",
    collection: "your-collection-name",
    limit: 10 // optional, defaults to 10
  }
})
```

### 3. Cross-Modal Search
Perform text-to-image or image-to-text search.

```typescript
use_mcp_tool({
  server_name: "jina-ai",
  tool_name: "cross_modal_search",
  arguments: {
    query: "a beautiful sunset", // or image URL for image2text
    mode: "text2image", // or "image2text"
    collection: "your-collection-name",
    limit: 10 // optional, defaults to 10
  }
})
```

## 📝 Response Format

All search tools return results in the following format:

```typescript
{
  content: [
    {
      type: "text",
      text: JSON.stringify({
        results: [
          {
            id: string,
            score: number,
            data: Record<string, any>
          }
        ]
      }, null, 2)
    }
  ]
}
```

## 🔐 Error Handling

The server handles various error cases:
- Invalid API key
- Missing or invalid parameters
- API rate limits
- Network errors
- Invalid collection names

All errors are properly formatted and returned with appropriate error codes and messages.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Jina AI](https://jina.ai/) for their excellent neural search platform
- [Model Context Protocol](https://github.com/modelcontextprotocol/protocol) for the MCP specification
