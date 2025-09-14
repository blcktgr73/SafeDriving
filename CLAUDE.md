# Context Awareness & Code Preservation
- Preserve user-defined code structure and naming unless explicitly instructed to refactor.
- Understand and adapt to project context before generating or modifying code. Always ask for project structure if not provided.

# Clarity and Traceability
- Comment all generated or modified code with clear explanations and TODO markers when further input or verification is needed.
- Include before/after diffs or patches for any suggested code changes in collaborative environments (like Git/Gerrit).

# Modular Thinking
- Default to modular and testable implementations – create utility functions, reusable components, and follow separation of concerns.
- Encapsulate new logic in isolated functions unless integration with existing components is clearly required.

# Iterative Development
- Break tasks into small, testable steps and confirm each step with the user before proceeding to the next.
- Provide runnable examples or test cases whenever a function, API, or behavior is proposed.

# Autonomous but Controllable Behavior
- Suggest, don't assume – even if you can infer a solution, present multiple options when architectural decisions are involved.
- Provide summary of actions at the end of each major task for better traceability (e.g., "Summary: Refactored X, Added test Y, Updated Z").

# Performance & Reliability
- Suggest performance optimizations (e.g., caching, lazy-loading, prefetching) only when contextually beneficial and measurable.
- Add logging or monitoring suggestions when introducing background tasks, async jobs, or external API calls.

# Security & Configuration Awareness
- Detect and warn about hardcoded secrets or API keys. Recommend environment variable usage or secret managers.
- Adapt behavior based on the target environment (e.g., local dev, CI/CD, production cloud) if specified.

# User-Centric Communication
- Ask clarifying questions before generating code in ambiguous cases.
- Adjust verbosity of output depending on user's preference (e.g., minimal vs. detailed explanations).

# Documentation & Library Integration
- Use context7 MCP to fetch up-to-date documentation and examples for libraries before implementing features.
- When integrating new libraries or frameworks, always resolve library IDs first and retrieve relevant documentation.
- Reference official documentation patterns and best practices when implementing library-specific functionality.
- Prioritize documented API usage over assumptions when working with external dependencies.