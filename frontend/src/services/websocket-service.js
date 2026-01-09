/*
 Contributors:
 - Maksym
 - Ahmet
*/


export class WebSocketService {
    constructor(url, options = {}) {
        this.url = url;
        this.ws = null;
        
        this.maxReconnectAttempts = options.maxReconnectAttempts;
        this.reconnectDelay = options.reconnectDelay;
        this.autoReconnect = options.autoReconnect;
        
        this.reconnectAttempts = 0;
        this.reconnectTimeout = null;
        this.shouldReconnect = false;
        
        this.messageHandlers = new Set();
        this.connectionHandlers = new Set();
        this.errorHandlers = new Set();
        this.closeHandlers = new Set();
    }

    connect() {
        if (this.ws && (this.ws.readyState === WebSocket.OPEN || this.ws.readyState === WebSocket.CONNECTING)) {
            console.log("WebSocket already connected or connecting");
            return;
        }

        this.shouldReconnect = this.autoReconnect;
        console.log("Connecting to WebSocket:", this.url);

        try {
            this.ws = new WebSocket(this.url);
            this.setupEventHandlers();
        } catch (error) {
            console.error("Error creating WebSocket:", error);
            this.notifyError(error);
        }
    }

    setupEventHandlers() {
        if (!this.ws) return;

        this.ws.onopen = () => {
            console.log("WebSocket connected");
            this.reconnectAttempts = 0;
            this.notifyConnection(true);
        };

        this.ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                this.notifyMessage(data);
            } catch (error) {
                console.error("Error parsing WebSocket message:", error);
                this.notifyError(error);
            }
        };

        this.ws.onclose = (event) => {
            console.log("WebSocket closed:", event.code, event.reason);
            this.notifyConnection(false);
            this.notifyClose(event);
            this.handleReconnect();
        };

        this.ws.onerror = (error) => {
            console.error("WebSocket error:", error);
            this.notifyError(new Error("WebSocket connection failed"));
        };
    }

    handleReconnect() {
        if (!this.shouldReconnect) return;
        if (this.reconnectAttempts >= this.maxReconnectAttempts) {
            console.log("Max reconnection attempts reached");
            return;
        }

        this.reconnectAttempts++;
        console.log(`Reconnection attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts}`);

        this.reconnectTimeout = setTimeout(() => {
            this.connect();
        }, this.reconnectDelay);
    }

    disconnect() {
        this.shouldReconnect = false;
        
        if (this.reconnectTimeout) {
            clearTimeout(this.reconnectTimeout);
            this.reconnectTimeout = null;
        }

        if (this.ws) {
            this.ws.close();
            this.ws = null;
        }
    }

    send(message) {
        if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
            console.warn("WebSocket is not connected");
            return false;
        }

        try {
            const payload = JSON.stringify(message)
            this.ws.send(payload);
            return true;
        } catch (error) {
            console.error("Error sending message:", error);
            this.notifyError(error);
            return false;
        }
    }

    onMessage(handler) {
        this.messageHandlers.add(handler);
        return () => this.messageHandlers.delete(handler);
    }

    onConnectionChange(handler) {
        this.connectionHandlers.add(handler);
        return () => this.connectionHandlers.delete(handler);
    }

    onError(handler) {
        this.errorHandlers.add(handler);
        return () => this.errorHandlers.delete(handler);
    }

    onClose(handler) {
        this.closeHandlers.add(handler);
        return () => this.closeHandlers.delete(handler);
    }

    get isConnected() {
        return this.ws?.readyState === WebSocket.OPEN;
    }

    get readyState() {
        return this.ws?.readyState ?? WebSocket.CLOSED;
    }

    notifyMessage(data) {
        this.messageHandlers.forEach(handler => {
            try {
                handler(data);
            } catch (error) {
                console.error("Error in message handler:", error);
            }
        });
    }

    notifyConnection(isConnected) {
        this.connectionHandlers.forEach(handler => {
            try {
                handler(isConnected);
            } catch (error) {
                console.error("Error in connection handler:", error);
            }
        });
    }

    notifyError(error) {
        this.errorHandlers.forEach(handler => {
            try {
                handler(error);
            } catch (error) {
                console.error("Error in error handler:", error);
            }
        });
    }

    notifyClose(event) {
        this.closeHandlers.forEach(handler => {
            try {
                handler(event);
            } catch (error) {
                console.error("Error in close handler:", error);
            }
        });
    }

    destroy() {
        this.disconnect();
        this.messageHandlers.clear();
        this.connectionHandlers.clear();
        this.errorHandlers.clear();
        this.closeHandlers.clear();
    }
}

export function createWebSocketService(url, options = {}) {
    return new WebSocketService(url, {
        maxReconnectAttempts: options.maxReconnectAttempts,
        reconnectDelay: options.reconnectDelay,
        autoReconnect: options.autoReconnect
    });
}

