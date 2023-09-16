FROM node:20-bullseye

# The workspace will be mounted under /workspace



RUN apt update -y && apt install -y curl

# Download and install Go
ENV GO_VERSION 1.20.3
RUN curl -L -o go.tar.gz https://golang.org/dl/go$GO_VERSION.linux-amd64.tar.gz &&     tar -C /usr/local -xzf go.tar.gz &&     rm go.tar.gz

# Set the Go environment variables
ENV GOPATH /go
ENV PATH $PATH:/usr/local/go/bin:$GOPATH/bin

# Verify the installation
RUN go version
