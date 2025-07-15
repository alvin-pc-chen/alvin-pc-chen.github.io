---
layout: post
title: I Vibe Coded an App in a Language I've Never Used—Here’s What I Learned
date: 2025-06-25 21:01:00
description: How I learned to stop worrying and love the singularity.
tags: ai coding, HCI
categories: exploration
thumbnail: assets/img/blog/ai-coding/profile.png
---

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/blog/ai-coding/intro.webp" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    The promise of AI coding tools is seductive: write a prompt, get working code. But how close are we to that vision today?
</div>

I was an early adopter of GitHub Copilot, and while the code suggestions have been great I never bought into vibe coding. To cut through the hype, I decided to try a few popular AI code editors and evaluate their capabilities. Since I specifically wanted to see how much of the development process these tools could automate, I chose to develop an app in Swift, a language I've never used before. The results were enlightening-while I still don't believe in "vibe coding", I now incorporate AI a lot more into my workflow.

Why it matters: These tools are transforming how we code—but also exposing critical questions about interface design, task coverage, and infrastructure reliability.

Purpose: In this post, I’ll share what worked, what didn’t, and what I believe the next generation of AI developer tools should focus on.

### Key Takeaways

- AI code editors are very good at generating code and fixing common errors, building simple apps and writing scripts is accessible to anyone willing to try.
- Performance gets progressively worse as the codebase increases in size and complexity; don't be afraid of rolling back changes when the AI starts chasing its own tail.
- Code generation works well for custom situations but many frameworks (Django, Flask, React, Xcode projects, etc.) have built-in methods for starting new projects. Good editors prioritize using these capabilities instead of generating boilerplate to ensure that the initial codebase works.

### The Experiment
Context: I tested several AI coding IDEs using a language I’m not familiar with to explore their capabilities and limitations.

The question I wanted to answer was straightforward: how much of the development process can an AI code editor automate? The app I had in mind was a simple journaling app that would help me collect my thoughts by forcing me to write something every two hours. I already had prior experience with web app development tools like Django and React.js, but in order for my app to initiate journaling sessions, I needed it to interface directly with my system. 

stuff about each tool sampled

generating ideas, prompting, testing

### What Worked Well

🔨 

🧠 3 Big Challenges I’d Love to Tackle

Bootstrapping projects: Most tools excelled at setting up scaffolding and explaining unfamiliar boilerplate.

Natural language prompts: High-quality autocompletion and context-aware suggestions reduced guesswork.

Iterative refinement: In some cases, chat-based editing made it feel like pair programming with a tireless teammate.

### Where AI Coding Tools Still Struggle

Context fragility: Tools lost track of intent or required repeating information across prompts.

Multi-step task planning: Larger tasks (e.g., database setup + front-end wiring) often required manual orchestration.

Debugging black box: Hard to tell why things didn’t work—limited introspection into generated logic.

Cross-file editing: Editing across multiple files or systems (e.g., package managers, build configs) remained clunky.

### Are AI code editors just GPT wrappers?

I also tried to develop my app just using ChatGPT, to get a feel for the impact that context awareness and interface integration has on the development process. My hypothesis is that if the LLM contributes the vast majority of the performance, then there isn't much value add to a dedicated AI code editor. 

I use Copilot for suggestions on a daily basis and will continue to do so as long as it remains free, but I would likely opt-out of a subscription service and  

### General Design Recommendations for AI Editors

### Three Key Challenges Facing the Industry
1. How do we build code generation tools that solve real developer tasks?

    Current gap: Many tools can generate snippets, but few understand goals or full workflows.

    Opportunity: Move from “autocomplete on steroids” to “task-oriented systems” that track dependencies, side effects, and steps.

    Ideas: Systems that remember intent across sessions, support design-to-code workflows, or incorporate code review best practices.

2. What’s the next interface after autocomplete and chat?

    Problem: Autocomplete is passive; chat is linear.

    Speculation: Could we build a “visual planner” for code? A canvas for wiring components? Semantic search across the whole project?

    Future vision: Interfaces that combine conversation, diagrams, context-aware UI, and real-time previews.

3. How do we ensure reliability and scale across languages, IDEs, and devices?

    Observations: Even top-tier tools struggled with certain compilers, OS-level configs, or edge IDEs.

    Infrastructure challenge: Need for robust telemetry, fast sandboxing, and multi-platform testing to ensure consistent quality.

    Personal idea: Modular architecture that abstracts away environment-specific quirks while learning from telemetry data.

Section 5: Reflections and Takeaways

    Key insight: These tools are incredibly promising—but to become everyday assistants, they need to evolve from reactive code generators into proactive collaborators.

    Personal growth: Testing these tools made me think more deeply about UX in dev environments and the invisible complexity of toolchains.

    Career interest: I’m excited to contribute to teams solving these exact problems—building smarter, more robust tools that understand and empower developers at scale.

Call to Action

    For engineers: Try building in an unfamiliar language using an AI IDE—you’ll quickly see the magic and the gaps.

    For companies: Let’s talk if you’re working on the future of developer tooling. I’d love to help bridge the gap between ambition and real-world usage.

Would you like help drafting the full blog post based on this outline, or tailoring it for a specific company (e.g., Replit, Cursor, GitHub, Sourcegraph, etc.)?


<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/8.jpg" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/10.jpg" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>


<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/11.jpg" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/12.jpg" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/7.jpg" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
