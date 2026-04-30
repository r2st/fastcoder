# Security Policy

## Supported versions

We provide security fixes for the **latest minor release** on the `main` branch. Older versions are not supported.

| Version | Supported          |
| ------- | ------------------ |
| 3.1.x   | :white_check_mark: |
| < 3.1   | :x:                |

## Reporting a vulnerability

**Please do not open a public GitHub issue for security vulnerabilities.**

Instead, report the issue privately via one of:

1. **GitHub Security Advisories** — open a draft advisory on this repository (preferred).
2. **Email** — send a detailed report to the maintainer (see project contact in `pyproject.toml`).

Please include:

- A description of the vulnerability and its impact
- Steps to reproduce (PoC if possible)
- Affected versions / commits
- Any suggested mitigation

You can expect:

- Acknowledgement within **72 hours**
- An initial assessment within **7 days**
- A coordinated disclosure timeline once the fix is in progress

## Security expectations

`fastcoder` executes shell commands, writes code, and operates against your repositories — treat the running process as **trusted** and isolate it accordingly:

- Run inside a container or sandbox in production
- Set `AGENT_KEY_ENCRYPTION_KEY` to enable at-rest encryption of provider keys
- Restrict `AGENT_CORS_ORIGINS` to known frontends
- Place behind authenticated reverse-proxy if exposed beyond `localhost`
- Rotate the auto-generated super-admin password on first login

## Out of scope

- Vulnerabilities requiring physical access to the host
- Self-XSS in the admin panel
- Issues affecting only unsupported / forked versions
- Denial of service via resource exhaustion when no rate limit was configured

Thank you for helping keep `fastcoder` and its users safe.
