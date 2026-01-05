import { createContext, useContext, useEffect, useState, useCallback, useRef } from "react";
import { createWebSocketService } from "../services/websocket-service";

const WebSocketContext = createContext(null);

export function WebSocketProvider({ children, url, options = {} }) {
    const [isConnected, setIsConnected] = useState(false);
    const [lastMessage, setLastMessage] = useState(null);
    const [error, setError] = useState(null);
    
    const serviceRef = useRef(null);

    useEffect(() => {
        const service = createWebSocketService(url, options);
        serviceRef.current = service;

        const unsubConnection = service.onConnectionChange((connected) => {
            setIsConnected(connected);
        });

        const unsubMessage = service.onMessage((data) => {
            setLastMessage(data);
        });

        const unsubError = service.onError((err) => {
            setError(err.message || "WebSocket error");
        });

        service.connect();

        return () => {
            unsubConnection();
            unsubMessage();
            unsubError();
            service.destroy();
        };
    }, [url]);

    const connect = useCallback(() => {
        serviceRef.current?.connect();
    }, []);

    const disconnect = useCallback(() => {
        serviceRef.current?.disconnect();
    }, []);

    const send = useCallback((message) => {
        return serviceRef.current?.send(message) ?? false;
    }, []);


    const value = {
        isConnected,
        lastMessage,
        error,
        connect,
        disconnect,
        send,
        service: serviceRef.current,
    };

    return (
        <WebSocketContext.Provider value={value}>
            {children}
        </WebSocketContext.Provider>
    );
}

export function useWebSocket() {
    const context = useContext(WebSocketContext);
    
    if (!context) {
        throw new Error("useWebSocket must be used within a WebSocketProvider");
    }
    
    return context;
}