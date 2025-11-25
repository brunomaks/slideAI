FROM node:24-slim

WORKDIR /app

# Copy package files and install dependencies
COPY package*.json ./
RUN npm install

# Copy the rest of the code
COPY . .

# Expose Vite's default port
EXPOSE 5173

# Start development server with host exposure
CMD ["npm", "run", "dev", "--", "--host"]
