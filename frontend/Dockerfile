FROM node:18-alpine

WORKDIR /app

# Install dependencies
COPY package*.json ./
RUN npm install

# Copy source code
COPY . .

# Expose Vite's default port
EXPOSE 5173

# Start Vite dev server and expose it to the local network
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]
