# opencode 2, a demo project and Python, for trying prelude as a coding agent (see README.md). What the model
# runs stays in this container: docker-compose.yml gives it no route to the internet.
FROM python:3.13-slim
ARG TARGETARCH
ARG OPENCODE_VERSION=2.0.20
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl git ripgrep \
    && rm -rf /var/lib/apt/lists/*
# The release binary, checked against the sha256 of opencode's Homebrew formula (anomalyco/tap).
RUN case "$TARGETARCH" in \
      arm64) f=opencode-linux-arm64.tar.gz; sum=33ba758e484c82149de8d153305ef0c49539f8fa87e8ead076662e745a89272f ;; \
      *) f=opencode-linux-x64-baseline.tar.gz; sum=6b6fa605be3653d96d7b956fbd004b6520b36ae3f4adb63c561846ed9a65f3e6 ;; \
    esac \
    && curl -fsSL -o /tmp/opencode.tar.gz "https://opencode.ai/files/bin/${OPENCODE_VERSION}/${f}" \
    && echo "${sum}  /tmp/opencode.tar.gz" | sha256sum -c - \
    && tar -xzf /tmp/opencode.tar.gz -C /usr/local/bin opencode && rm /tmp/opencode.tar.gz
RUN useradd --create-home dev
USER dev
# opencode's user config (the mini-lab provider), so the project itself stays untouched.
RUN mkdir -p /home/dev/.config/opencode
COPY --chown=dev opencode.json /home/dev/.config/opencode/opencode.json
WORKDIR /home/dev/project
COPY --chown=dev project/ ./
RUN git init -q && git add -A && git -c user.name=dev -c user.email=dev@localhost commit -qm "Demo project"
CMD ["opencode"]
