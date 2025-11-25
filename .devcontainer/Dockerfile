FROM mcr.microsoft.com/devcontainers/anaconda:1-3

# Optionally skip the (often slow) conda env update during image build by passing
# build-arg SKIP_CONDA_UPDATE=true. This helps when you just want the container up
# quickly and install/adjust packages interactively later.
ARG SKIP_CONDA_UPDATE="false"

# Copy environment.yml (if found) to a temp location so we update the environment. Also
# copy "noop.txt" so the COPY instruction does not fail if no environment.yml exists.
COPY environment.yml* .devcontainer/noop.txt /tmp/conda-tmp/
RUN if [ "${SKIP_CONDA_UPDATE}" = "true" ]; then \
            echo "SKIP_CONDA_UPDATE=true: skipping conda env update"; \
        else \
            if [ -f "/tmp/conda-tmp/environment.yml" ]; then \
                umask 0002 && /opt/conda/bin/conda env update -n base -f /tmp/conda-tmp/environment.yml; \
            fi; \
        fi \
        && rm -rf /tmp/conda-tmp

# [Optional] Uncomment this section to install additional OS packages.
# RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
#     && apt-get -y install --no-install-recommends <your-package-list-here>
