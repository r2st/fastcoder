const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  Header, Footer, AlignmentType, LevelFormat, HeadingLevel,
  BorderStyle, WidthType, ShadingType, PageNumber, PageBreak,
  ExternalHyperlink, TableOfContents, TabStopType, TabStopPosition,
} = require("docx");

// ── Colors ──
const BRAND = "1A56DB";
const BRAND_LIGHT = "E8F0FE";
const ACCENT = "059669";
const ACCENT_LIGHT = "ECFDF5";
const WARN = "D97706";
const WARN_LIGHT = "FFF7ED";
const CRIT = "DC2626";
const CRIT_LIGHT = "FEF2F2";
const GRAY = "6B7280";
const GRAY_LIGHT = "F3F4F6";
const BORDER_COLOR = "D1D5DB";
const TABLE_HEADER = "1E3A5F";
const TABLE_HEADER_TEXT = "FFFFFF";
const NEW_TAG = "059669";
const UPDATED_TAG = "D97706";

// ── Helpers ──
const border = { style: BorderStyle.SINGLE, size: 1, color: BORDER_COLOR };
const borders = { top: border, bottom: border, left: border, right: border };
const cellMargins = { top: 80, bottom: 80, left: 120, right: 120 };

const PAGE_WIDTH = 12240;
const MARGIN = 1440;
const CONTENT_WIDTH = PAGE_WIDTH - 2 * MARGIN; // 9360

function heading(text, level) {
  return new Paragraph({
    heading: level,
    spacing: { before: level === HeadingLevel.HEADING_1 ? 360 : 240, after: 120 },
    children: [new TextRun({ text, bold: true, font: "Arial", size: level === HeadingLevel.HEADING_1 ? 32 : level === HeadingLevel.HEADING_2 ? 26 : 22 })],
  });
}

function para(text, opts = {}) {
  return new Paragraph({
    spacing: { after: 120 },
    ...opts,
    children: Array.isArray(text) ? text : [new TextRun({ text, font: "Arial", size: 22, ...opts.run })],
  });
}

function bold(text) { return new TextRun({ text, bold: true, font: "Arial", size: 22 }); }
function normal(text) { return new TextRun({ text, font: "Arial", size: 22 }); }
function colored(text, color) { return new TextRun({ text, font: "Arial", size: 22, color }); }
function tag(text, color) { return new TextRun({ text: ` [${text}] `, bold: true, font: "Arial", size: 18, color }); }
function italic(text) { return new TextRun({ text, italics: true, font: "Arial", size: 22 }); }

function headerCell(text, width) {
  return new TableCell({
    borders, width: { size: width, type: WidthType.DXA },
    shading: { fill: TABLE_HEADER, type: ShadingType.CLEAR },
    margins: cellMargins,
    verticalAlign: "center",
    children: [new Paragraph({ children: [new TextRun({ text, bold: true, font: "Arial", size: 20, color: TABLE_HEADER_TEXT })] })],
  });
}

function cell(content, width, opts = {}) {
  const children = Array.isArray(content)
    ? content.map(c => typeof c === "string" ? new Paragraph({ children: [new TextRun({ text: c, font: "Arial", size: 20 })] }) : c)
    : [new Paragraph({ children: [new TextRun({ text: content, font: "Arial", size: 20, ...opts.run })] })];
  return new TableCell({
    borders, width: { size: width, type: WidthType.DXA },
    shading: opts.shading || { fill: "FFFFFF", type: ShadingType.CLEAR },
    margins: cellMargins,
    children,
  });
}

function bulletItem(textRuns) {
  return new Paragraph({
    numbering: { reference: "bullets", level: 0 },
    spacing: { after: 60 },
    children: Array.isArray(textRuns) ? textRuns : [normal(textRuns)],
  });
}

function subBullet(textRuns) {
  return new Paragraph({
    numbering: { reference: "subbullets", level: 0 },
    spacing: { after: 40 },
    children: Array.isArray(textRuns) ? textRuns : [normal(textRuns)],
  });
}

function numberedItem(textRuns, ref = "numbers") {
  return new Paragraph({
    numbering: { reference: ref, level: 0 },
    spacing: { after: 60 },
    children: Array.isArray(textRuns) ? textRuns : [normal(textRuns)],
  });
}

function calloutBox(title, body, fillColor) {
  return new Table({
    width: { size: CONTENT_WIDTH, type: WidthType.DXA },
    columnWidths: [CONTENT_WIDTH],
    rows: [new TableRow({ children: [
      new TableCell({
        borders: { top: { style: BorderStyle.SINGLE, size: 3, color: fillColor === CRIT_LIGHT ? CRIT : fillColor === WARN_LIGHT ? WARN : ACCENT },
          bottom: border, left: { style: BorderStyle.SINGLE, size: 3, color: fillColor === CRIT_LIGHT ? CRIT : fillColor === WARN_LIGHT ? WARN : ACCENT }, right: border },
        width: { size: CONTENT_WIDTH, type: WidthType.DXA },
        shading: { fill: fillColor, type: ShadingType.CLEAR },
        margins: { top: 120, bottom: 120, left: 200, right: 200 },
        children: [
          new Paragraph({ spacing: { after: 60 }, children: [new TextRun({ text: title, bold: true, font: "Arial", size: 22, color: fillColor === CRIT_LIGHT ? CRIT : fillColor === WARN_LIGHT ? WARN : ACCENT })] }),
          new Paragraph({ children: [new TextRun({ text: body, font: "Arial", size: 20 })] }),
        ],
      }),
    ] })],
  });
}

function spacer() { return new Paragraph({ spacing: { after: 80 }, children: [] }); }

// ── Build the document ──
const doc = new Document({
  styles: {
    default: { document: { run: { font: "Arial", size: 22 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 32, bold: true, font: "Arial", color: "111827" },
        paragraph: { spacing: { before: 360, after: 200 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 26, bold: true, font: "Arial", color: "1F2937" },
        paragraph: { spacing: { before: 280, after: 160 }, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 22, bold: true, font: "Arial", color: "374151" },
        paragraph: { spacing: { before: 200, after: 120 }, outlineLevel: 2 } },
    ],
  },
  numbering: {
    config: [
      { reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "\u2022", alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
      { reference: "subbullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "\u25E6", alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 1200, hanging: 360 } } } }] },
      { reference: "numbers", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
      { reference: "numbers2", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
      { reference: "numbers3", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
      { reference: "numbers4", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
      { reference: "numbers5", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
    ],
  },
  sections: [{
    properties: {
      page: {
        size: { width: PAGE_WIDTH, height: 15840 },
        margin: { top: MARGIN, right: MARGIN, bottom: MARGIN, left: MARGIN },
      },
    },
    headers: {
      default: new Header({ children: [
        new Paragraph({
          border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: BRAND, space: 1 } },
          spacing: { after: 0 },
          tabStops: [{ type: TabStopType.RIGHT, position: TabStopPosition.MAX }],
          children: [
            new TextRun({ text: "Autonomous Dev Agent", bold: true, font: "Arial", size: 18, color: BRAND }),
            new TextRun({ text: "\tRequirements Specification v4.0", font: "Arial", size: 18, color: GRAY }),
          ],
        }),
      ] }),
    },
    footers: {
      default: new Footer({ children: [
        new Paragraph({
          border: { top: { style: BorderStyle.SINGLE, size: 4, color: BORDER_COLOR, space: 1 } },
          tabStops: [{ type: TabStopType.RIGHT, position: TabStopPosition.MAX }],
          children: [
            new TextRun({ text: "Confidential", font: "Arial", size: 16, color: GRAY }),
            new TextRun({ text: "\tPage ", font: "Arial", size: 16, color: GRAY }),
            new TextRun({ children: [PageNumber.CURRENT], font: "Arial", size: 16, color: GRAY }),
          ],
        }),
      ] }),
    },
    children: [

      // ── TITLE PAGE ──
      spacer(), spacer(), spacer(), spacer(),
      new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 40 }, children: [
        new TextRun({ text: "AUTONOMOUS SOFTWARE DEVELOPMENT AGENT", bold: true, font: "Arial", size: 40, color: BRAND }),
      ] }),
      new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 }, children: [
        new TextRun({ text: "Product Requirements Specification", font: "Arial", size: 28, color: GRAY }),
      ] }),
      new Paragraph({ alignment: AlignmentType.CENTER, border: { bottom: { style: BorderStyle.SINGLE, size: 8, color: BRAND, space: 1 } }, spacing: { after: 300 }, children: [] }),
      spacer(),
      new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 80 }, children: [
        new TextRun({ text: "Version 4.0  \u2014  Updated with Market Research Findings", font: "Arial", size: 22, color: "374151" }),
      ] }),
      new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 80 }, children: [
        new TextRun({ text: "April 2026", font: "Arial", size: 22, color: GRAY }),
      ] }),
      new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 80 }, children: [
        new TextRun({ text: "Author: Suman", font: "Arial", size: 22, color: GRAY }),
      ] }),
      spacer(), spacer(),
      calloutBox(
        "Research-Driven Update",
        "This revision incorporates findings from the Market Analysis of the Production-Grade Autonomous Dev Agent Gap (ChatGPT Deep Research, April 2026), DORA AI Tensions report, METR SWE-bench maintainer study, and competitive analysis of GitHub Copilot Cloud Agent, Lovable, Claude Code, Cursor, and Sourcegraph. New and updated requirements are tagged throughout.",
        BRAND_LIGHT,
      ),

      new Paragraph({ children: [new PageBreak()] }),

      // ── TABLE OF CONTENTS ──
      heading("Table of Contents", HeadingLevel.HEADING_1),
      new TableOfContents("Table of Contents", { hyperlink: true, headingStyleRange: "1-3" }),
      new Paragraph({ children: [new PageBreak()] }),

      // ══════════════════════════════════════════════════════════════
      // SECTION 1: EXECUTIVE SUMMARY
      // ══════════════════════════════════════════════════════════════
      heading("1. Executive Summary", HeadingLevel.HEADING_1),
      para([
        normal("The Autonomous Software Development Agent accepts user stories as input and autonomously performs the full software development lifecycle: analysis, planning, code generation, testing, deployment, verification, and self-improvement. It operates on "),
        bold("existing production codebases"),
        normal(", not just greenfield projects, closing the key market gap identified in research."),
      ]),
      spacer(),
      calloutBox(
        "Key Market Insight",
        "METR research (2026) found that roughly half of AI-generated pull requests that passed SWE-bench automated grading would still not be merged by maintainers, due to code quality, repo standards, or risks not captured by automated checks. This validates the core thesis: test-passing is necessary but not sufficient. The product must produce merge-ready evidence, not just passing tests.",
        WARN_LIGHT,
      ),
      spacer(),
      para([
        normal("The agent is positioned as a "),
        bold("configurable autonomy + evidence system"),
        normal(" rather than a better model. It spans developer productivity, testing automation, and DevOps/ALM governance \u2014 a convergence product targeting the USD 7.37B AI code tools market (forecast USD 23.97B by 2030) and the adjacent USD 25.43B automation testing market."),
      ]),

      // ══════════════════════════════════════════════════════════════
      // SECTION 2: GOALS & DESIGN PRINCIPLES
      // ══════════════════════════════════════════════════════════════
      new Paragraph({ children: [new PageBreak()] }),
      heading("2. Goals & Design Principles", HeadingLevel.HEADING_1),
      heading("2.1 Primary Goals", HeadingLevel.HEADING_2),
      numberedItem([bold("Story-to-Software: "), normal("Accept a user story and produce working, tested, deployed, merge-ready software with evidence bundles.")]),
      numberedItem([bold("Autonomous Iteration: "), normal("Detect failures and fix them without human intervention, with progressive context enrichment.")]),
      numberedItem([bold("Self-Improvement: "), normal("Learn from past successes and failures; reduce iteration cycles over time via post-mortem learning loops.")]),
      numberedItem([bold("Multi-Model Flexibility: "), normal("Route tasks to different LLMs based on capability, cost, and SLA envelopes.")]),
      numberedItem([bold("Existing Codebase Support: "), tag("NEW", NEW_TAG), normal("Work on multi-year, multi-repo production systems \u2014 not limited to greenfield.")]),
      numberedItem([bold("Production-Grade Reliability: "), tag("NEW", NEW_TAG), normal("Deliver outcomes that pass not just automated tests but also maintainer-quality standards.")]),
      numberedItem([bold("Controlled Speed: "), tag("NEW", NEW_TAG), normal("Enterprise buyers purchase controlled speed, not raw speed. Verification and governance are first-class.")]),

      heading("2.2 Design Principles", HeadingLevel.HEADING_2),
      new Table({
        width: { size: CONTENT_WIDTH, type: WidthType.DXA },
        columnWidths: [2800, 6560],
        rows: [
          new TableRow({ children: [headerCell("Principle", 2800), headerCell("Description", 6560)] }),
          ...[
            ["Loop-First", "Every action feeds back into an evaluation loop. The agent converges, never just finishes."],
            ["Fail-Safe by Default", "Sandboxed environments. Destructive actions require explicit approval gates."],
            ["Observable", "Every decision, code change, and deployment is logged with full provenance for audit."],
            ["Model-Agnostic", "The LLM layer is abstracted. Models can be swapped, mixed, or A/B tested."],
            ["Incremental Delivery", "Small increments: commit, test, and deploy after each coherent change."],
            ["Human Override", "Configurable approval gates at any pipeline stage. Policy-based, not all-or-nothing."],
            ["Bounded Autonomy", "Oversight is architectural: permissioning, escalation policies, post-deployment monitoring."],
            ["Machine-Checkable Specs", "Acceptance criteria compiled into executable tests (BDD) before code generation."],
            ["Evidence-First [NEW]", "Every merge request includes an evidence bundle: test reports, SAST results, coverage, change impact analysis. The system produces merge-ready proof, not just code."],
            ["Policy-as-Code [NEW]", "Quality gates defined as first-class policy objects (not hardcoded). Industry and regulatory templates supported."],
          ].map(([p, d]) => new TableRow({ children: [
            cell(p, 2800, { run: { bold: true }, shading: { fill: GRAY_LIGHT, type: ShadingType.CLEAR } }),
            cell(d, 6560),
          ] })),
        ],
      }),

      // ══════════════════════════════════════════════════════════════
      // SECTION 3: CLOSED-LOOP QUALITY ENFORCEMENT
      // ══════════════════════════════════════════════════════════════
      new Paragraph({ children: [new PageBreak()] }),
      heading("3. Closed-Loop Quality Enforcement", HeadingLevel.HEADING_1),
      para([tag("NEW SECTION", NEW_TAG), normal(" Derived from research finding that verification is the primary bottleneck, not code generation.")]),
      spacer(),
      calloutBox(
        "Research Driver",
        "DORA reports that time saved in code creation is re-allocated to auditing and verification rather than shipping net-new value. Higher AI adoption correlates with both higher throughput AND higher instability. Closed-loop quality enforcement directly attacks this verification tax.",
        ACCENT_LIGHT,
      ),
      spacer(),

      heading("3.1 Quality Gates as First-Class Policy Objects", HeadingLevel.HEADING_2),
      para("Quality gates are not hardcoded checks but configurable policy objects that run automatically as part of the agent loop. Each gate produces structured evidence for merge decisions."),
      new Table({
        width: { size: CONTENT_WIDTH, type: WidthType.DXA },
        columnWidths: [2200, 3000, 2100, 2060],
        rows: [
          new TableRow({ children: [headerCell("Gate", 2200), headerCell("Description", 3000), headerCell("Default Threshold", 2100), headerCell("Configurable", 2060)] }),
          ...[
            ["Lint", "Code style and formatting checks", "Zero errors", "Yes \u2014 per-project"],
            ["Type Check", "Static type analysis (mypy, tsc)", "Zero errors", "Yes \u2014 strict/basic"],
            ["Unit Tests", "Unit test suite execution", "\u226580% coverage", "Yes \u2014 min coverage %"],
            ["Integration Tests", "Cross-module integration tests", "All passing", "Yes \u2014 skip list"],
            ["E2E Tests", "End-to-end acceptance tests", "All passing", "Yes \u2014 scope"],
            ["SAST", "Static Application Security Testing", "Zero high/critical", "Yes \u2014 severity"],
            ["Dependency Audit", "Known vulnerability scanning", "Zero critical CVEs", "Yes \u2014 severity"],
            ["Secret Detection", "Detect leaked secrets/keys", "Zero findings", "No \u2014 always enforced"],
            ["Coverage Delta", "Coverage must not decrease", "\u22650% delta", "Yes \u2014 threshold"],
            ["Performance Budget", "Bundle size / latency limits", "Per-project", "Yes \u2014 limits"],
            ["Migration Safety", "DB migration reversibility check", "All reversible", "Yes \u2014 allow list"],
          ].map(([g, d, t, c]) => new TableRow({ children: [
            cell(g, 2200, { run: { bold: true } }),
            cell(d, 3000),
            cell(t, 2100),
            cell(c, 2060),
          ] })),
        ],
      }),
      spacer(),

      heading("3.2 Evidence Bundle Generation", HeadingLevel.HEADING_2),
      para("Every PR/merge request produced by the agent must include a structured evidence bundle:"),
      bulletItem([bold("Test Report: "), normal("Pass/fail counts, coverage percentage, failure details")]),
      bulletItem([bold("SAST Report: "), normal("Security scan results with severity classifications")]),
      bulletItem([bold("Change Impact Analysis: "), normal("Files changed, dependency graph impact, blast radius")]),
      bulletItem([bold("Acceptance Criteria Map: "), normal("Each BDD criterion linked to its verifying test(s)")]),
      bulletItem([bold("Cost Summary: "), normal("Tokens used, model calls, total cost for this story")]),
      bulletItem([bold("Reviewer Checklist: "), normal("Pre-filled checklist of what the agent verified and what needs human review")]),

      heading("3.3 DevSecOps Integration", HeadingLevel.HEADING_2),
      para([
        normal("Research found "),
        bold("meaningful rates of security weaknesses"),
        normal(" in AI-generated code across Python and JavaScript (CWE categories). Quality gates "),
        bold("must"),
        normal(" include DevSecOps checks, not only tests. Security scanning runs automatically at every iteration, not just before deployment."),
      ]),

      // ══════════════════════════════════════════════════════════════
      // SECTION 4: CODEBASE INTELLIGENCE
      // ══════════════════════════════════════════════════════════════
      new Paragraph({ children: [new PageBreak()] }),
      heading("4. Codebase Intelligence Engine", HeadingLevel.HEADING_1),
      para([tag("UPDATED", UPDATED_TAG), normal(" Expanded based on research identifying context at scale as a hard unsolved problem.")]),
      spacer(),

      heading("4.1 Multi-Layer Indexing", HeadingLevel.HEADING_2),
      new Table({
        width: { size: CONTENT_WIDTH, type: WidthType.DXA },
        columnWidths: [2100, 4200, 3060],
        rows: [
          new TableRow({ children: [headerCell("Index Layer", 2100), headerCell("What It Captures", 4200), headerCell("Use Case", 3060)] }),
          ...[
            ["AST Index", "Abstract syntax trees for every source file", "Signatures, hierarchies, available APIs"],
            ["Dependency Graph", "Import relationships, module boundaries, circulars", "Impact analysis across files"],
            ["Semantic Embeddings", "Vector embeddings of code chunks", "Natural language search over code"],
            ["Symbol Table", "All exported symbols, types, locations, usage", "Auto-import, name collision avoidance"],
            ["Convention Profile", "Detected naming, structure, error handling, test patterns", "Style-consistent code generation"],
            ["API Surface Map", "HTTP endpoints, CLI commands, event handlers", "Understanding external interfaces"],
            ["Ownership Map [NEW]", "CODEOWNERS, git blame, review frequency", "Route approvals to correct humans"],
            ["Build/Test Recipe [NEW]", "Build commands, test invocations, CI config per repo", "Deterministic build/test execution"],
            ["Cross-Repo Index [NEW]", "Shared types, contracts, API schemas across repos", "Multi-repo change safety"],
          ].map(([l, w, u]) => new TableRow({ children: [
            cell(l, 2100, { run: { bold: true } }),
            cell(w, 4200),
            cell(u, 3060),
          ] })),
        ],
      }),
      spacer(),

      heading("4.2 Existing Codebase Import", HeadingLevel.HEADING_2),
      para([
        tag("NEW", NEW_TAG),
        normal("Unlike Lovable (export-only GitHub integration), this agent must import and operate on existing repositories. The codebase intelligence engine performs a first-run analysis that builds a complete ProjectProfile within minutes, not hours."),
      ]),
      bulletItem([bold("Git clone + branch checkout: "), normal("Support SSH and HTTPS with credential management")]),
      bulletItem([bold("Full indexing on import: "), normal("AST, dependency graph, convention scan, symbol table")]),
      bulletItem([bold("Incremental re-indexing: "), normal("Only changed files and their dependents after each code generation step")]),
      bulletItem([bold("Multi-repo support: "), normal("Index shared types and API contracts across microservice repositories")]),

      // ══════════════════════════════════════════════════════════════
      // SECTION 5: CONFIGURABLE AUTONOMY
      // ══════════════════════════════════════════════════════════════
      new Paragraph({ children: [new PageBreak()] }),
      heading("5. Configurable Autonomy & Human-in-the-Loop", HeadingLevel.HEADING_1),
      para([tag("NEW SECTION", NEW_TAG), normal(" Derived from the autonomy/safety tension documented by GitHub and Anthropic.")]),
      spacer(),
      calloutBox(
        "Research Driver",
        "GitHub documents that by default, Actions workflows will NOT run automatically when Copilot pushes changes to a PR, because workflows may access privileged secrets. This illustrates the core tension: platforms prevent fully closed-loop CI by default. The solution is policy-based intervention points, not all-or-nothing autonomy.",
        WARN_LIGHT,
      ),
      spacer(),

      heading("5.1 Policy-Based Intervention Points", HeadingLevel.HEADING_2),
      para("Instead of binary human-in-the-loop, the system offers configurable intervention policies:"),
      new Table({
        width: { size: CONTENT_WIDTH, type: WidthType.DXA },
        columnWidths: [2600, 4300, 2460],
        rows: [
          new TableRow({ children: [headerCell("Policy Rule", 2600), headerCell("Trigger Condition", 4300), headerCell("Action", 2460)] }),
          ...[
            ["Auto-Merge", "All gates pass AND change criticality < threshold", "Merge without human review"],
            ["Review Required", "Change touches infra/workflow/auth files", "Pause for human approval"],
            ["Security Sign-off", "SAST findings > 0 OR auth-related changes", "Route to security reviewer"],
            ["Budget Gate", "Story cost exceeds per-story or daily budget", "Pause and notify operator"],
            ["Ambiguity Gate", "Analyzer detects unclear requirements", "Request clarification from submitter"],
            ["Criticality Escalation", "Change blast radius > N files or touches core module", "Require senior review"],
          ].map(([p, t, a]) => new TableRow({ children: [
            cell(p, 2600, { run: { bold: true } }),
            cell(t, 4300),
            cell(a, 2460),
          ] })),
        ],
      }),
      spacer(),

      heading("5.2 Approval Gate Configuration", HeadingLevel.HEADING_2),
      para("Approval gates are defined in the project configuration and can be customized per-repository:"),
      bulletItem([bold("pre_code: "), normal("Review execution plan before code generation begins")]),
      bulletItem([bold("pre_deploy: "), normal("Review changes and evidence bundle before deployment")]),
      bulletItem([bold("pre_production: "), normal("Additional approval for production (vs staging) deployment")]),
      bulletItem([bold("budget_exceeded: "), normal("Triggered when cost limits are hit")]),
      bulletItem([bold("ambiguity_detected: "), normal("Triggered when story analysis finds unclear requirements")]),
      bulletItem([bold("blast_radius: "), tag("NEW", NEW_TAG), normal("Triggered when change impact exceeds configured threshold")]),
      bulletItem([bold("security_finding: "), tag("NEW", NEW_TAG), normal("Triggered when SAST detects any finding above configured severity")]),

      heading("5.3 Audit Trail & Governance", HeadingLevel.HEADING_2),
      para([
        tag("NEW", NEW_TAG),
        normal("Enterprise governance requires answering: who approved what, when, and why. Every decision point is logged with:"),
      ]),
      bulletItem("Actor (agent, human reviewer name, policy rule that auto-approved)"),
      bulletItem("Timestamp and story context"),
      bulletItem("Evidence that was available at the time of the decision"),
      bulletItem("The specific gate/policy that was evaluated"),
      bulletItem("SSO/SCIM integration for enterprise identity management"),

      // ══════════════════════════════════════════════════════════════
      // SECTION 6: MULTI-MODEL ROUTING
      // ══════════════════════════════════════════════════════════════
      new Paragraph({ children: [new PageBreak()] }),
      heading("6. Multi-Model Routing & Cost Optimization", HeadingLevel.HEADING_1),
      para([tag("UPDATED", UPDATED_TAG), normal(" Expanded from basic routing to a full routing+budget engine per research recommendations.")]),

      heading("6.1 Task-Purpose Routing", HeadingLevel.HEADING_2),
      new Table({
        width: { size: CONTENT_WIDTH, type: WidthType.DXA },
        columnWidths: [2200, 2600, 2400, 2160],
        rows: [
          new TableRow({ children: [headerCell("Task Purpose", 2200), headerCell("Recommended Tier", 2600), headerCell("Rationale", 2400), headerCell("Fallback", 2160)] }),
          ...[
            ["Story Analysis", "Mid-tier", "Structured extraction", "Top-tier"],
            ["Planning", "Top-tier", "Requires deep reasoning", "Mid-tier + retry"],
            ["Code Generation", "Top-tier", "Quality-critical", "Mid-tier for simple"],
            ["Test Generation", "Mid-tier", "Pattern-based", "Top-tier for complex"],
            ["Code Review", "Top-tier (different provider)", "Independent perspective", "Same provider"],
            ["Security Scan", "Specialized SAST model [NEW]", "Domain expertise", "Top-tier + SAST tool"],
            ["Context Retrieval", "Low-tier / Embeddings [NEW]", "High-volume, low-cost", "Mid-tier"],
            ["Error Diagnosis", "Mid-tier", "Pattern matching", "Top-tier on retry 3+"],
          ].map(([t, r, ra, f]) => new TableRow({ children: [
            cell(t, 2200, { run: { bold: true } }),
            cell(r, 2600),
            cell(ra, 2400),
            cell(f, 2160),
          ] })),
        ],
      }),
      spacer(),

      heading("6.2 Cost Budget Engine", HeadingLevel.HEADING_2),
      para([tag("NEW", NEW_TAG), normal(" Predictable cost and SLA envelopes per task type:")]),
      bulletItem([bold("Per-story budget: "), normal("Default $5.00, configurable. Agent pauses when exceeded.")]),
      bulletItem([bold("Per-day budget: "), normal("Default $100.00. Prevents runaway costs from batch processing.")]),
      bulletItem([bold("Per-month budget: "), normal("Default $2,000.00. Hard cap with admin notification.")]),
      bulletItem([bold("Cost-per-token tracking: "), normal("Real-time cost accumulation across all model calls per story.")]),
      bulletItem([bold("Model comparison: "), normal("Ability to run same prompt across multiple models and compare quality/cost.")]),
      bulletItem([bold("SLA envelopes: "), normal("Configurable max latency per task type; auto-fallback on timeout.")]),

      // ══════════════════════════════════════════════════════════════
      // SECTION 7: ERROR RECOVERY WITH LEARNING
      // ══════════════════════════════════════════════════════════════
      new Paragraph({ children: [new PageBreak()] }),
      heading("7. Error Recovery & Post-Mortem Learning", HeadingLevel.HEADING_1),
      para([tag("UPDATED", UPDATED_TAG), normal(" Enhanced with post-mortem learning loops from research recommendations.")]),

      heading("7.1 Error Taxonomy (Unchanged)", HeadingLevel.HEADING_2),
      para("Eight error categories with SHA-256 fingerprinting, oscillation detection, and progressive context enrichment (Attempt 1\u20135 strategy). See System Design v3.1 Section 9 for full specification."),

      heading("7.2 Post-Mortem Learning Loops", HeadingLevel.HEADING_2),
      para([tag("NEW", NEW_TAG), normal(" When a gate fails or a reviewer requests changes, the agent performs automated post-mortem:")]),
      numberedItem([bold("Classify the failure mode: "), normal("Map to the error taxonomy and determine root cause category")], "numbers2"),
      numberedItem([bold("Update repo-specific heuristics: "), normal("If a convention violation or repo-specific pattern caused failure, update the ProjectProfile")], "numbers2"),
      numberedItem([bold("Adjust future plans/gates: "), normal("If a quality gate consistently fails for a pattern, pre-emptively include that check in planning")], "numbers2"),
      numberedItem([bold("Store as memory entry: "), normal("Add to the Memory Store with effectiveness score for RAG retrieval in future stories")], "numbers2"),
      numberedItem([bold("Reviewer feedback integration: "), normal("When a human rejects or requests changes, capture the specific concern and train the agent to pre-empt it")], "numbers2"),
      spacer(),
      calloutBox(
        "Closing the METR Gap",
        "The METR study found maintainers reject AI-generated PRs for reasons beyond test failure: code quality, repo standards, naming conventions. Post-mortem learning directly addresses this by turning each rejection into a repo-specific heuristic that prevents the same rejection class in future stories.",
        ACCENT_LIGHT,
      ),

      heading("7.3 Memory Architecture", HeadingLevel.HEADING_2),
      para("Five-tier memory system (working, episodic, semantic, procedural, project) with Jaccard similarity search and consolidation after story completion:"),
      bulletItem([bold("Working Memory: "), normal("Current story context, active iteration state")]),
      bulletItem([bold("Episodic Memory: "), normal("Error fixes and their outcomes per story")]),
      bulletItem([bold("Semantic Memory: "), normal("General patterns and conventions learned across stories")]),
      bulletItem([bold("Procedural Memory: "), normal("Build/test/deploy recipes that worked for specific repo types")]),
      bulletItem([bold("Project Memory: "), normal("Repo-specific conventions, reviewer preferences, known failure patterns")]),
      spacer(),
      para([
        tag("NEW", NEW_TAG),
        bold("Memory Validity Tracking: "),
        normal("Memories must be invalidated when branches and code evolve. Each memory entry is linked to a git commit SHA and is flagged for re-evaluation when the referenced code changes."),
      ]),

      // ══════════════════════════════════════════════════════════════
      // SECTION 8: PR-NATIVE WORKFLOW
      // ══════════════════════════════════════════════════════════════
      new Paragraph({ children: [new PageBreak()] }),
      heading("8. PR-Native Workflow & CI/CD Integration", HeadingLevel.HEADING_1),
      para([tag("NEW SECTION", NEW_TAG), normal(" Derived from research on packaging and distribution strategy.")]),

      heading("8.1 PR as the Delivery Unit", HeadingLevel.HEADING_2),
      para("The agent delivers changes as pull requests with full evidence bundles, integrating with existing branch protection and required status checks:"),
      bulletItem([bold("Feature branch per story: "), normal("Conventional naming (feature/STORY-{id}-{slug})")]),
      bulletItem([bold("Conventional commits: "), normal("feat/fix/refactor prefix matching story type")]),
      bulletItem([bold("Evidence bundle attached: "), normal("Test reports, SAST results, coverage diffs, change impact as PR comments or check annotations")]),
      bulletItem([bold("Branch protection integration: "), normal("Respect required reviewers, required checks, and merge restrictions")]),
      bulletItem([bold("PR description generation: "), normal("Auto-generated summary with acceptance criteria checklist")]),

      heading("8.2 CI/CD Pipeline Integration", HeadingLevel.HEADING_2),
      bulletItem([bold("GitHub Actions / GitLab CI: "), normal("Trigger external CI pipeline and wait for results before proceeding")]),
      bulletItem([bold("Status check alignment: "), normal("Agent's quality gates reported as GitHub status checks")]),
      bulletItem([bold("Secrets management: "), normal("Never expose secrets in PR descriptions, logs, or generated code. Integrate with vault/secrets manager.")]),
      bulletItem([bold("Rollback automation: "), normal("If post-deploy verification fails, automatic rollback with incident report")]),

      heading("8.3 Multi-Repository Operations", HeadingLevel.HEADING_2),
      para([
        tag("NEW", NEW_TAG),
        normal("GitHub Copilot Cloud Agent cannot make changes across multiple repositories in one run. This agent supports coordinated cross-repo changes:"),
      ]),
      bulletItem("Detect shared types/contracts that span repositories"),
      bulletItem("Create synchronized PRs across affected repos"),
      bulletItem("Validate cross-repo contract compatibility before merging any PR"),
      bulletItem("Configurable scope: single-repo (default) or multi-repo (requires explicit configuration)"),

      // ══════════════════════════════════════════════════════════════
      // SECTION 9: SECURITY
      // ══════════════════════════════════════════════════════════════
      new Paragraph({ children: [new PageBreak()] }),
      heading("9. Security Requirements", HeadingLevel.HEADING_1),
      para([tag("UPDATED", UPDATED_TAG), normal(" Expanded based on research highlighting security weaknesses in AI-generated code.")]),

      heading("9.1 API & Authentication Security", HeadingLevel.HEADING_2),
      bulletItem([bold("Bearer token authentication: "), normal("All API routes protected by auto-generated or configured token")]),
      bulletItem([bold("Constant-time comparison: "), normal("Token validation uses timing-safe comparison")]),
      bulletItem([bold("API key in-memory only: "), normal("LLM provider keys stored in-memory, never written to disk or logs")]),
      bulletItem([bold("Input validation: "), normal("Size limits on all request bodies; Pydantic validation on all models")]),
      bulletItem([bold("CORS restricted: "), normal("Localhost-only by default; configurable via environment variable")]),

      heading("9.2 Execution Security", HeadingLevel.HEADING_2),
      bulletItem([bold("Shell command allowlist: "), normal("Only pre-approved commands can execute. Unknown commands rejected.")]),
      bulletItem([bold("Shell injection prevention: "), normal("All user-derived values passed through shlex.quote(); URL validation before curl")]),
      bulletItem([bold("Environment isolation: "), normal("Blocked env vars (LD_PRELOAD, PYTHONPATH, etc.) can never be overridden")]),
      bulletItem([bold("Working directory enforcement: "), normal("All commands constrained to project directory")]),
      bulletItem([bold("Resource limits: "), normal("CPU time (5 min), memory (2 GB), disk (10 GB) per command")]),

      heading("9.3 Generated Code Security", HeadingLevel.HEADING_2),
      para([
        tag("NEW", NEW_TAG),
        normal("Empirical research found meaningful rates of CWE vulnerabilities in AI-generated code. Every code generation pass includes:"),
      ]),
      bulletItem("Automatic SAST scan before code review stage"),
      bulletItem("Secret detection (regex + entropy) on all generated files"),
      bulletItem("Dependency audit for any newly added packages"),
      bulletItem("SQL injection and XSS pattern detection for web code"),
      bulletItem("Self-reflection security axis score \u2265 0.8 required"),

      heading("9.4 Information Disclosure Prevention", HeadingLevel.HEADING_2),
      bulletItem("Error responses never expose internal details; use request IDs for log correlation"),
      bulletItem("API keys masked in all admin panel responses"),
      bulletItem("Stack traces logged server-side only, never returned to client"),
      bulletItem("Request body size limit (1 MB) to prevent JSON bomb attacks"),

      // ══════════════════════════════════════════════════════════════
      // SECTION 10: COMPETITIVE DIFFERENTIATION
      // ══════════════════════════════════════════════════════════════
      new Paragraph({ children: [new PageBreak()] }),
      heading("10. Competitive Differentiation Matrix", HeadingLevel.HEADING_1),
      para([tag("NEW SECTION", NEW_TAG), normal(" Derived from research competitive analysis.")]),
      spacer(),
      new Table({
        width: { size: CONTENT_WIDTH, type: WidthType.DXA },
        columnWidths: [1900, 1300, 1300, 1300, 1300, 2160],
        rows: [
          new TableRow({ children: [
            headerCell("Capability", 1900), headerCell("Lovable", 1300), headerCell("Claude Code", 1300),
            headerCell("Copilot Agent", 1300), headerCell("Cursor", 1300), headerCell("This Agent", 2160),
          ] }),
          ...[
            ["Existing repo import", "No", "Yes", "Yes", "Yes", "Yes + multi-repo"],
            ["Closed-loop testing", "On request", "Yes", "Yes", "No", "Enforced by default"],
            ["Quality gates", "None", "None", "CI hooks", "None", "11 configurable gates"],
            ["Security scanning", "None", "None", "Via CI", "None", "Built-in SAST"],
            ["Evidence bundles", "None", "None", "Partial", "None", "Full per-PR"],
            ["Multi-model routing", "No", "No", "Yes", "Yes", "Yes + cost budget"],
            ["Post-mortem learning", "None", "Auto-memory", "Memory (preview)", "None", "5-tier + feedback"],
            ["Cross-repo changes", "No", "No", "No", "No", "Yes"],
            ["Configurable autonomy", "None", "Basic", "Admin config", "None", "Policy-based gates"],
            ["Audit trail", "Basic", "None", "Actions logs", "None", "Full governance"],
          ].map(([cap, lov, claude, copilot, cursor, agent]) => new TableRow({ children: [
            cell(cap, 1900, { run: { bold: true } }),
            cell(lov, 1300),
            cell(claude, 1300),
            cell(copilot, 1300),
            cell(cursor, 1300),
            cell(agent, 2160, { shading: { fill: ACCENT_LIGHT, type: ShadingType.CLEAR } }),
          ] })),
        ],
      }),

      // ══════════════════════════════════════════════════════════════
      // SECTION 11: IMPLEMENTATION PRIORITIES
      // ══════════════════════════════════════════════════════════════
      new Paragraph({ children: [new PageBreak()] }),
      heading("11. Implementation Priority & Roadmap", HeadingLevel.HEADING_1),
      para([tag("UPDATED", UPDATED_TAG), normal(" Reordered based on research-identified market wedges.")]),
      spacer(),

      heading("11.1 Phase 1: Foundation (Current \u2014 Complete)", HeadingLevel.HEADING_2),
      para("Core orchestrator state machine, component wiring, adapter layer, all 13 components initialized."),
      bulletItem("Orchestrator with 10-state FSM and convergence detection"),
      bulletItem("Story Analyzer, Planner, Code Generator, Test Generator, Code Reviewer"),
      bulletItem("Error taxonomy with 8 categories and SHA-256 fingerprinting"),
      bulletItem("Multi-model router with 4 providers (Anthropic, OpenAI, Gemini, Ollama)"),
      bulletItem("Admin panel and workspace UI"),
      bulletItem("Security hardening (auth, injection prevention, CORS, env isolation)"),

      heading("11.2 Phase 2: Quality Enforcement (Next \u2014 Highest Priority)", HeadingLevel.HEADING_2),
      para([tag("UPDATED", UPDATED_TAG), normal(" Research identifies this as the most monetizable wedge.")]),
      numberedItem([bold("SAST integration: "), normal("Bandit (Python), ESLint security plugin (JS), Semgrep")], "numbers3"),
      numberedItem([bold("Evidence bundle generation: "), normal("Structured reports attached to each PR")], "numbers3"),
      numberedItem([bold("Quality gate policy engine: "), normal("Configurable gate definitions, thresholds, enforcement levels")], "numbers3"),
      numberedItem([bold("Secret detection: "), normal("Regex + entropy scanning on all generated code")], "numbers3"),
      numberedItem([bold("Coverage delta enforcement: "), normal("Coverage must not decrease with new changes")], "numbers3"),

      heading("11.3 Phase 3: Codebase Intelligence (High Priority)", HeadingLevel.HEADING_2),
      numberedItem([bold("Existing repo import workflow: "), normal("Git clone, full index, ProjectProfile generation")], "numbers4"),
      numberedItem([bold("Cross-repo dependency tracking: "), normal("Shared types, API contracts, schema compatibility")], "numbers4"),
      numberedItem([bold("Ownership map: "), normal("CODEOWNERS parsing, git blame analysis, review routing")], "numbers4"),
      numberedItem([bold("Incremental re-indexing: "), normal("File-hash based change detection with dependency-aware invalidation")], "numbers4"),

      heading("11.4 Phase 4: Enterprise Governance (Medium Priority)", HeadingLevel.HEADING_2),
      numberedItem([bold("Policy-based intervention engine: "), normal("Configurable rules for auto-merge, escalation, security sign-off")], "numbers5"),
      numberedItem([bold("Audit trail: "), normal("Complete decision log with actor, timestamp, evidence, gate evaluated")], "numbers5"),
      numberedItem([bold("SSO/SCIM integration: "), normal("Enterprise identity management for approval workflows")], "numbers5"),
      numberedItem([bold("Post-mortem learning loops: "), normal("Automated failure classification, heuristic updates, memory storage")], "numbers5"),
      numberedItem([bold("Memory validity tracking: "), normal("Git SHA-linked memories with auto-invalidation on code changes")], "numbers5"),

      // ══════════════════════════════════════════════════════════════
      // SECTION 12: ACCEPTANCE CRITERIA
      // ══════════════════════════════════════════════════════════════
      new Paragraph({ children: [new PageBreak()] }),
      heading("12. Success Metrics & Acceptance Criteria", HeadingLevel.HEADING_1),
      para([tag("NEW SECTION", NEW_TAG), normal(" Measurable targets derived from research benchmarks.")]),
      spacer(),
      new Table({
        width: { size: CONTENT_WIDTH, type: WidthType.DXA },
        columnWidths: [3100, 2800, 3460],
        rows: [
          new TableRow({ children: [headerCell("Metric", 3100), headerCell("Target", 2800), headerCell("Research Basis", 3460)] }),
          ...[
            ["PR merge rate by maintainers", "\u226575% (vs METR 50%)", "METR SWE-bench maintainer study"],
            ["Quality gate pass rate (first attempt)", "\u226560%", "Reduce verification tax per DORA"],
            ["Mean iterations to convergence", "\u22643 iterations", "System design target from v3.1"],
            ["Security vulnerability rate", "<5% of generated code", "Copilot empirical security study"],
            ["Time from story to merge-ready PR", "<30 minutes (median)", "Competitive with human developers"],
            ["Cost per story (median)", "<$3.00", "Cost optimization engine target"],
            ["Post-mortem learning retention", "\u226580% fewer same-class rejections", "Feedback loop effectiveness"],
            ["Existing codebase onboarding", "<5 minutes for full index", "Import workflow requirement"],
          ].map(([m, t, r]) => new TableRow({ children: [
            cell(m, 3100, { run: { bold: true } }),
            cell(t, 2800),
            cell(r, 3460, { run: { italics: true } }),
          ] })),
        ],
      }),

      // ══════════════════════════════════════════════════════════════
      // APPENDIX: RESEARCH SOURCES
      // ══════════════════════════════════════════════════════════════
      new Paragraph({ children: [new PageBreak()] }),
      heading("Appendix: Research Sources", HeadingLevel.HEADING_1),
      para("The following research sources informed the requirements updates in this document:"),
      spacer(),
      bulletItem([bold("DORA 2025/2026: "), italic("Balancing AI Tensions: Moving from AI adoption to effective SDLC use"), normal(" \u2014 dora.dev")]),
      bulletItem([bold("METR 2026: "), italic("Many SWE-bench-Passing PRs Would Not Be Merged into Main"), normal(" \u2014 metr.org")]),
      bulletItem([bold("Copilot Security Study: "), italic("Security Weaknesses of Copilot-Generated Code in GitHub Projects: An Empirical Study"), normal(" \u2014 arxiv.org/html/2310.02059v3")]),
      bulletItem([bold("CSET Georgetown: "), italic("Cybersecurity Risks of AI-Generated Code"), normal(" \u2014 cset.georgetown.edu")]),
      bulletItem([bold("GitHub Copilot Cloud Agent: "), italic("About GitHub Copilot Cloud Agent"), normal(" \u2014 docs.github.com")]),
      bulletItem([bold("GitHub Copilot Memory: "), italic("Building an Agentic Memory System for GitHub Copilot"), normal(" \u2014 github.blog")]),
      bulletItem([bold("Anthropic: "), italic("Measuring AI Agent Autonomy in Practice"), normal(" \u2014 anthropic.com/research")]),
      bulletItem([bold("Lovable Documentation: "), italic("Welcome to Lovable + Test and Verify + GitHub Integration"), normal(" \u2014 docs.lovable.dev")]),
      bulletItem([bold("Sourcegraph: "), italic("Agentic Chat \u2014 Think Twice, Answer Once"), normal(" \u2014 sourcegraph.com/blog")]),
      bulletItem([bold("Cursor: "), italic("Best Practices for Coding with Agents"), normal(" \u2014 cursor.com/blog")]),
      bulletItem([bold("Market Data: "), italic("AI Code Tools Market Size & 2030 Trends Report"), normal(" \u2014 mordorintelligence.com")]),
      bulletItem([bold("Market Data: "), italic("Automation Testing Market Size, Share, Trends Report 2030"), normal(" \u2014 grandviewresearch.com")]),
      bulletItem([bold("SlashData: "), italic("There are 47.2 Million Developers in the World"), normal(" \u2014 slashdata.co")]),
    ],
  }],
});

Packer.toBuffer(doc).then(buffer => {
  fs.writeFileSync("/sessions/dazzling-practical-shannon/mnt/auto-agent/auto-dev-agent/docs/REQUIREMENTS_v4.docx", buffer);
  console.log("REQUIREMENTS_v4.docx created successfully");
});
