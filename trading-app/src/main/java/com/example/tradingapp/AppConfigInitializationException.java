package com.example.tradingapp;

public class AppConfigInitializationException extends RuntimeException {
    public AppConfigInitializationException(String message) {
        super(message);
    }

    public AppConfigInitializationException(String message, Throwable cause) {
        super(message, cause);
    }
}
