# syntax=docker/dockerfile:1
#
# Pinned dual-host test image for pi-openrouter-advanced-provider.
#
# Pinned toolchain, checksum-verified at build time against the official
# release checksum files (a mismatch fails the build):
#   base image  oven/bun:1.4.3-debian
#   pi  1.5.0   earendil-works/pi release tarball pi-linux-x64.tar.gz
#   omp 18.16.0 can1357/oh-my-pi release binary omp-linux-x64
#
# The repo is COPYed to /app so CI builds are deterministic. For local runs,
# scripts/container-test.mjs bind-mounts the checkout read-only at /repo and
# runs the scenario from there; dependencies then resolve from the host
# checkout's node_modules (mounted read-only when present), otherwise they
# fall back to the image-installed copy — symlinked at /node_modules so
# module resolution from any repo location finds it.
#
# Runtime contract (set exclusively by scripts/container-test.mjs):
#   SCENARIO_HOSTS         "pi", "omp", or "pi,omp"
#   SCENARIO_REPO_DIR      repo root inside the container (pi/, test/, scripts/, package.json)
#   SCENARIO_HOME          writable isolated HOME (tmpfs); hosts' config,
#                          auth stores and session files live only here
#   SCENARIO_ARTIFACTS_DIR where the scenario writes the mock NDJSON log and
#                          other artifacts (inside SCENARIO_HOME)
#   SCENARIO_FIXTURE_KEY   bearer key the mock accepts (sk-or-test-container)
# The mock OpenRouter server runs inside the container and the container runs
# with --network=none. No credentials are copied in or mounted.

FROM oven/bun:1.4.3-debian

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
	&& apt-get install -y --no-install-recommends ca-certificates curl \
	&& rm -rf /var/lib/apt/lists/*

ARG PI_VERSION=1.5.0
# sha256 of pi-linux-x64.tar.gz from the v1.5.0 SHA256SUMS release asset.
ARG PI_LINUX_X64_SHA256=df870ba0e344cb7b3358d8d8f19397b8b8d225a8669784736ad68a5c2169d53b
ARG OMP_VERSION=18.16.0
# sha256 of omp-linux-x64 from the v18.16.0 SHA256SUMS.txt release asset.
ARG OMP_LINUX_X64_SHA256=6498ed785bab00d2926e0387e0ec02362d6d45f1c3d0d1b57c1618b7f95c9264

# pi: official linux x64 release tarball. Verify against the pinned checksum
# AND the live release SHA256SUMS asset, then extract the compiled binary.
RUN curl -fsSL -o /tmp/pi-linux-x64.tar.gz \
		"https://github.com/earendil-works/pi/releases/download/v${PI_VERSION}/pi-linux-x64.tar.gz" \
	&& echo "${PI_LINUX_X64_SHA256}  /tmp/pi-linux-x64.tar.gz" | sha256sum -c - \
	&& curl -fsSL \
		"https://github.com/earendil-works/pi/releases/download/v${PI_VERSION}/SHA256SUMS" \
		| grep -F "${PI_LINUX_X64_SHA256}" >/dev/null \
	&& mkdir -p /opt/pi \
	&& tar -xzf /tmp/pi-linux-x64.tar.gz -C /opt/pi \
	&& chmod +x /opt/pi/pi/pi \
	&& ln -s /opt/pi/pi/pi /usr/local/bin/pi \
	&& rm /tmp/pi-linux-x64.tar.gz

# omp: official linux x64 release binary, verified the same way.
RUN curl -fsSL -o /usr/local/bin/omp \
		"https://github.com/can1357/oh-my-pi/releases/download/v${OMP_VERSION}/omp-linux-x64" \
	&& echo "${OMP_LINUX_X64_SHA256}  /usr/local/bin/omp" | sha256sum -c - \
	&& curl -fsSL \
		"https://github.com/can1357/oh-my-pi/releases/download/v${OMP_VERSION}/SHA256SUMS.txt" \
		| grep -F "${OMP_LINUX_X64_SHA256}" >/dev/null \
	&& chmod +x /usr/local/bin/omp

WORKDIR /app
COPY . /app/
RUN bun install \
	&& ln -s /app/node_modules /node_modules

# Default working directory: /app (CI layout). The runner overrides the
# command; this CMD only explains how the image is meant to be used.
CMD ["/bin/sh", "-c", "echo 'Use scripts/container-test.mjs (npm run test:container) to drive this image.'; echo 'Env contract: SCENARIO_HOSTS SCENARIO_REPO_DIR SCENARIO_HOME SCENARIO_ARTIFACTS_DIR SCENARIO_FIXTURE_KEY'; exit 64"]
