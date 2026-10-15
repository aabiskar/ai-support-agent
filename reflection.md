# Project Reflection: Customer Support AI Agent

Building this customer support AI agent helped me understand how a single AI agent can use multiple tools and AWS services to handle e-commerce requests. Instead of relying only on a language model, the agent can retrieve knowledge, call external services, perform calculations, and interact with web pages through dedicated integrations.

One important integration was Amazon Bedrock AgentCore Gateway with MCP tools. I tested an API-based order-tracking target and a Lambda-based refund-processing target. During testing, I verified that the agent retrieved the actual total of $139.99 for order ORD-002 and passed that amount to the refund tool. This highlighted why agents must retrieve and validate business data before performing actions.

I also implemented retrieval-augmented generation (RAG) using the Amazon Bedrock Agent Runtime Retrieve API. CloudWatch logs confirmed that the deployed agent invoked `search_knowledge_base` when asked about Platinum loyalty benefits. This demonstrated how the agent can retrieve relevant knowledge rather than relying exclusively on its prompt.

Browser execution presented a deployment challenge involving Playwright's Node.js binary permissions. A workaround involved copying the binary to `/tmp/node` and setting `PLAYWRIGHT_NODEJS_PATH`. The Code Interpreter integration also encountered an expired AWS security token error, so successful execution of that integration remains unverified.

I improved the loyalty calculation to support standard, device, and fresh product categories, calculate earned points from the final total, and return a fallback response when Code Interpreter is unavailable. Tests passed for standard and device categories, including their expected final totals and points balances. Fresh-category behavior and actual fallback execution still require verification.

Persistent memory was another important learning area. After updating the memory namespace handling and testing with the same customer ID across separate sessions, the agent successfully recalled the customer's name and response preference.

For production, I would strengthen credential management, add automated integration tests and monitoring, and secure Gateway tools with appropriate authentication. In particular, a Gateway using the `NONE` authorizer should not expose refund operations publicly; a Cognito JWT authorizer would provide stronger access control.