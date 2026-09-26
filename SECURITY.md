# Security policy

mini-lab is an educational project, but its platform handles accounts, API keys and
(test-mode) payments, so we take security reports seriously.

## Reporting a vulnerability

Please **do not open a public issue**. Use GitHub's private vulnerability reporting
("Security" tab → "Report a vulnerability") with a description, the affected
component and, if possible, steps to reproduce. You should get an answer within a
week.

## Deploying mini-lab

It's built to run locally. If you expose it anyway:

- set a random `MINILAB_INTERNAL_TOKEN` (the services refuse to listen beyond
  localhost with the default one);
- serve the platform over HTTPS and set `MINILAB_PLATFORM_URL` accordingly;
- configure Stripe keys, or anyone can add free test-mode credits;
- keep the inference server (port 8001) off the public network.
