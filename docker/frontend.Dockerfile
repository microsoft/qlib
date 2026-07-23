# Build stage
FROM node:18-alpine as build

WORKDIR /app

# Copy package.json and package-lock.json
COPY package*.json ./

# Install dependencies
RUN npm install

# Copy application code
COPY . .

# Build the application
RUN npm run build

# Production stage - Use Node.js to serve static files
FROM node:18-alpine

WORKDIR /app

# Install a simple HTTP server
RUN npm install -g serve

# Copy built assets from build stage
COPY --from=build /app/dist /app/dist

# Expose port
EXPOSE 80

# Start the server
CMD ["serve", "-s", "dist", "-l", "80"]
