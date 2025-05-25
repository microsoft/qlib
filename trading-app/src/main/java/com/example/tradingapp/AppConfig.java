package com.example.tradingapp;

// Added for Javadoc for AppConfigInitializationException
// import com.example.tradingapp.AppConfigInitializationException; 

import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.util.Properties;

/**
 * Manages application configuration settings loaded from a properties file.
 * This class provides static methods to access configuration values such as API credentials,
 * URLs, and other application settings. It handles loading properties from a specified
 * file (defaulting to "config.properties") and includes error handling for missing
 * files or properties.
 * 
 * The configuration file is expected to be in the classpath, typically under `src/main/resources`.
 */
public class AppConfig {

    private static final Properties properties = new Properties();
    private static final String DEFAULT_CONFIG_FILE = "config.properties";

    static {
        // Load default properties on class initialization
        loadProperties(DEFAULT_CONFIG_FILE, true);
    }
    
    // Private constructor to prevent instantiation as this class provides only static methods.
    private AppConfig() { 
    }

    /**
     * Loads properties from the given filename located in the classpath.
     * This method is synchronized to prevent concurrent modification issues if called from multiple threads,
     * though typically it's called once at startup and then for tests.
     * 
     * @param filename The name of the properties file (e.g., "config.properties").
     * @param isCritical If true, an {@link AppConfigInitializationException} will be thrown if the file cannot be loaded.
     *                   If false, errors during loading will be logged to `System.err`, but the application/test
     *                   may continue (properties might be empty or defaults used if available).
     * @throws AppConfigInitializationException if `isCritical` is true and the file cannot be found or read.
     */
    public static synchronized void loadProperties(String filename, boolean isCritical) {
        properties.clear(); // Clear any existing properties before loading new ones
        try (InputStream input = AppConfig.class.getClassLoader().getResourceAsStream(filename)) {
            if (input == null) {
                String errorMessage = "Configuration file '" + filename + "' not found in classpath.";
                if (isCritical) {
                    System.err.println("FATAL ERROR: " + errorMessage 
                                     + "\nThe application cannot function without this configuration file."
                                     + "\nPlease ensure '" + filename + "' exists and is correctly placed (e.g., in src/main/resources or src/test/resources).");
                    throw new AppConfigInitializationException(errorMessage);
                } else {
                    System.err.println("WARNING: " + errorMessage + " (Non-critical, continuing...)");
                    // Not throwing an exception allows tests to proceed or application to use defaults if any were set.
                }
                return; // Return if input is null, properties will be empty or retain previous if not cleared.
            }
            properties.load(input);
            System.out.println("Successfully loaded properties from: " + filename);
        } catch (IOException ex) {
            String errorMessage = "Could not read configuration file '" + filename + "'. It might be corrupted or unreadable.";
            if (isCritical) {
                 System.err.println("FATAL ERROR: " + errorMessage);
                 ex.printStackTrace(System.err);
                throw new AppConfigInitializationException(errorMessage, ex);
            } else {
                System.err.println("WARNING: " + errorMessage + " (Non-critical, continuing...)");
                ex.printStackTrace(System.err);
            }
        }
    }


    /**
     * Retrieves a property value for the given key.
     * Throws a RuntimeException if the key is not found or the value is empty,
     * as these are considered critical configuration errors.
     * 
     * @param key The property key.
     * @return The property value.
     * @throws RuntimeException if the key is not found or the value is empty.
     */
    private static String getProperty(String key) {
        String value = properties.getProperty(key);
        if (value == null || value.trim().isEmpty()) {
            // This is a configuration error, a RuntimeException is appropriate.
            // The message will be caught by the global exception handler or handled by the caller.
            throw new RuntimeException("Configuration Error: Missing required value for key '" + key + "' in the loaded properties file (e.g., " + DEFAULT_CONFIG_FILE + ")");
        }
        return value;
    }
    
    /**
     * Retrieves a property value for the given key.
     * Returns an empty string if the key is not found or the value is empty.
     * Useful for optional properties.
     * 
     * @param key The property key.
     * @return The property value or an empty string if not found/empty.
     */
    private static String getPropertyOrEmpty(String key) {
        return properties.getProperty(key, "");
    }

    /** @return The User ID for API authentication. */
    public static String getUserId() {
        return getProperty("api.userID");
    }

    /** @return The Password for API authentication. */
    public static String getPassword() {
        return getProperty("api.password");
    }

    /** @return The PAN or DOB for API authentication. */
    public static String getPanOrDOB() {
        return getProperty("api.panOrDOB");
    }

    /** @return The Vendor ID for API authentication. */
    public static String getVendorId() {
        return getProperty("api.vendorId");
    }

    /** @return The Time-based One-Time Password (TOTP), if configured. Empty string otherwise. */
    public static String getTotp() {
        return getPropertyOrEmpty("api.totp");
    }

    /** @return The API Key for the application. */
    public static String getApiKey() {
        return getProperty("api.apiKey");
    }
    
    /** @return The Client Code, if configured. Empty string otherwise. */
    public static String getClientCode() {
        return getPropertyOrEmpty("api.clientCode");
    }

    /** @return The base URL for the UAT (User Acceptance Testing) API environment. */
    public static String getBaseUrlUat() {
        return getProperty("api.baseUrl.uat");
    }

    /** @return The base URL for the LIVE (Production) API environment. */
    public static String getBaseUrlLive() {
        return getProperty("api.baseUrl.live");
    }

    /** @return The current API environment setting ("uat" or "live"). */
    public static String getApiEnvironment() {
        return getProperty("api.environment");
    }

    /**
     * Determines and returns the appropriate base URL based on the 'api.environment' setting.
     * 
     * @return The base URL (either UAT or Live).
     * @throws RuntimeException if 'api.environment' is not set to "uat" or "live".
     */
    public static String getBaseUrl() {
        String environment = getApiEnvironment();
        if ("live".equalsIgnoreCase(environment)) {
            return getBaseUrlLive();
        } else if ("uat".equalsIgnoreCase(environment)) {
            // This ensures UAT URL is fetched if environment is UAT.
            return getBaseUrlUat();
        } else {
            // This condition indicates a misconfiguration in 'api.environment'.
            throw new RuntimeException("Configuration Error: Invalid value for 'api.environment' in " 
                                     + CONFIG_FILE + ". Found: '" + environment + "', but expected 'uat' or 'live'.");
        }
    }

    /** @return The Source ID for API requests (e.g., "WEB"). */
    public static String getSourceId() {
        return getProperty("api.sourceId");
    }

    /** @return The application browser name, used in API headers. */
    public static String getAppBrowserName() {
        return getProperty("app.browserName");
    }

    /** @return The application browser version, used in API headers. */
    public static String getAppBrowserVersion() {
        return getProperty("app.browserVersion");
    }
}
