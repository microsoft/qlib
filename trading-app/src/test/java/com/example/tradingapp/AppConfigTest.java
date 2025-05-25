package com.example.tradingapp;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.function.Executable;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.util.Properties;

import static org.junit.jupiter.api.Assertions.*;

public class AppConfigTest {

    private static final String TEST_CONFIG_FILE = "test_config.properties";
    private static final String DEFAULT_CONFIG_FILE = "config.properties"; // The actual one AppConfig loads by default
    private static Path tempDefaultConfigPath; // Path to temporary default config
    private static Path originalDefaultConfigPath; // Path to the original default config if it exists

    @BeforeAll
    static void setUpClass() {
        // This setup is tricky because AppConfig loads on class initialization via static block.
        // To test different scenarios like "file not found" or "missing keys",
        // we need to influence the file AppConfig tries to load *before* it loads.
        // The `loadProperties` method added for testability is the primary way to manage this.

        // For testing the default loading mechanism (if it were not for System.exit)
        // we might need to manipulate the actual "config.properties" in the classpath, which is risky.
        // The `loadProperties(filename, isCritical)` allows us to bypass the static block for most tests.
        System.out.println("AppConfigTest: Initializing test specific configurations.");
    }

    @AfterEach
    void tearDown() {
        // Reload the default properties after each test to ensure a clean state for subsequent tests
        // or other test classes, especially if a test modified the loaded properties.
        // Use isCritical=false for tearDown to avoid test failures if default is missing,
        // as some tests might intentionally test scenarios where default config is problematic.
        AppConfig.loadProperties(DEFAULT_CONFIG_FILE, false); // Or use a known good test file
    }

    @Test
    void testSuccessfulPropertyLoading() {
        AppConfig.loadProperties(TEST_CONFIG_FILE, true); // Load specific test properties

        assertEquals("testUser", AppConfig.getUserId());
        assertEquals("testPass", AppConfig.getPassword());
        assertEquals("TESTPAN", AppConfig.getPanOrDOB());
        assertEquals("testVendor", AppConfig.getVendorId());
        assertEquals("123456", AppConfig.getTotp());
        assertEquals("testApiKey", AppConfig.getApiKey());
        assertEquals("TCLIENT", AppConfig.getClientCode());
        assertEquals("https://test.uat.api", AppConfig.getBaseUrlUat());
        assertEquals("https://test.live.api", AppConfig.getBaseUrlLive());
        assertEquals("uat", AppConfig.getApiEnvironment());
        assertEquals("TESTWEB", AppConfig.getSourceId());
        assertEquals("TestBrowser", AppConfig.getAppBrowserName());
        assertEquals("1.0", AppConfig.getAppBrowserVersion());
    }

    @Test
    void testGetBaseUrl_UAT() {
        AppConfig.loadProperties(TEST_CONFIG_FILE, true);
        // Properties mock = new Properties();
        // mock.setProperty("api.environment", "uat");
        // mock.setProperty("api.baseUrl.uat", "https://uat.url");
        // AppConfig.setProperties(mock); // If we had a setter
        assertEquals("https://test.uat.api", AppConfig.getBaseUrl());
    }

    @Test
    void testGetBaseUrl_LIVE() {
        // Create a temporary properties file for this specific test case
        Properties liveProps = new Properties();
        liveProps.setProperty("api.environment", "live");
        liveProps.setProperty("api.baseUrl.live", "https://specific.live.url");
        liveProps.setProperty("api.baseUrl.uat", "https://specific.uat.url"); // Must be present due to getProperty
        // Add other mandatory fields to prevent getProperty from failing
        liveProps.setProperty("api.userID", "dummy");
        liveProps.setProperty("api.password", "dummy");
        liveProps.setProperty("api.panOrDOB", "dummy");
        liveProps.setProperty("api.vendorId", "dummy");
        liveProps.setProperty("api.apiKey", "dummy");
        liveProps.setProperty("api.sourceId", "dummy");
        liveProps.setProperty("app.browserName", "dummy");
        liveProps.setProperty("app.browserVersion", "dummy");


        String tempLiveConfigFile = "temp_live_config.properties";
        try {
            Path tempPath = Paths.get("target/test-classes/" + tempLiveConfigFile);
            Files.createDirectories(tempPath.getParent());
            try (OutputStream out = Files.newOutputStream(tempPath)) {
                liveProps.store(out, "Temporary live config");
            }
            AppConfig.loadProperties(tempLiveConfigFile, true);
            assertEquals("https://specific.live.url", AppConfig.getBaseUrl());
            Files.deleteIfExists(tempPath);
        } catch (IOException e) {
            fail("Failed to create or delete temporary config file for testGetBaseUrl_LIVE: " + e.getMessage());
        }
    }
    
    @Test
    void testMissingRequiredProperty() {
        // Create a properties file missing a required key
        Properties missingKeyProps = new Properties();
        missingKeyProps.setProperty("api.userID", "testUser");
        // api.password is missing
        missingKeyProps.setProperty("api.panOrDOB", "TESTPAN");
        // Add other mandatory fields to allow loading up to the point of the missing key.
        missingKeyProps.setProperty("api.vendorId", "testVendor");
        missingKeyProps.setProperty("api.apiKey", "testApiKey");
        missingKeyProps.setProperty("api.sourceId", "TESTWEB");
        missingKeyProps.setProperty("app.browserName", "TestBrowser");
        missingKeyProps.setProperty("app.browserVersion", "1.0");
        missingKeyProps.setProperty("api.environment", "uat");
        missingKeyProps.setProperty("api.baseUrl.uat", "https://test.uat.api");
        missingKeyProps.setProperty("api.baseUrl.live", "https://test.live.api");


        String tempMissingKeyFile = "temp_missing_key_config.properties";
        try {
            Path tempPath = Paths.get("target/test-classes/" + tempMissingKeyFile);
            Files.createDirectories(tempPath.getParent());
            try (OutputStream out = Files.newOutputStream(tempPath)) {
                missingKeyProps.store(out, "Temporary missing key config");
            }

            AppConfig.loadProperties(tempMissingKeyFile, true); // Load the incomplete properties

            Exception exception = assertThrows(RuntimeException.class, () -> {
                AppConfig.getPassword(); // This should trigger the exception
            });
            assertTrue(exception.getMessage().contains("Missing required value for key 'api.password'"));
            Files.deleteIfExists(tempPath);
        } catch (IOException e) {
            fail("Failed to create or delete temporary config file for testMissingRequiredProperty: " + e.getMessage());
        }
    }

    @Test
    void testGetPropertyOrEmpty_KeyExists() {
        AppConfig.loadProperties(TEST_CONFIG_FILE, true);
        assertEquals("123456", AppConfig.getTotp()); // Assuming api.totp is in test_config.properties
    }

    @Test
    void testGetPropertyOrEmpty_KeyMissing() {
         Properties emptyProps = new Properties();
        // Add other mandatory fields to prevent getProperty from failing during other calls if any
        emptyProps.setProperty("api.userID", "dummy");
        emptyProps.setProperty("api.password", "dummy");
        emptyProps.setProperty("api.panOrDOB", "dummy");
        emptyProps.setProperty("api.vendorId", "dummy");
        emptyProps.setProperty("api.apiKey", "dummy");
        emptyProps.setProperty("api.sourceId", "dummy");
        emptyProps.setProperty("app.browserName", "dummy");
        emptyProps.setProperty("app.browserVersion", "dummy");
        emptyProps.setProperty("api.environment", "uat");
        emptyProps.setProperty("api.baseUrl.uat", "https://test.uat.api");
        emptyProps.setProperty("api.baseUrl.live", "https://test.live.api");
        // api.clientCode is intentionally missing for this test of getPropertyOrEmpty

        String tempEmptyKeyFile = "temp_empty_key_config.properties";
        try {
            Path tempPath = Paths.get("target/test-classes/" + tempEmptyKeyFile);
             Files.createDirectories(tempPath.getParent());
            try (OutputStream out = Files.newOutputStream(tempPath)) {
                emptyProps.store(out, "Temporary empty key config");
            }
            AppConfig.loadProperties(tempEmptyKeyFile, true);
            assertEquals("", AppConfig.getClientCode()); // Should return empty string
            Files.deleteIfExists(tempPath);
        } catch (IOException e) {
            fail("Failed to create or delete temporary config file for testGetPropertyOrEmpty_KeyMissing: " + e.getMessage());
        }
    }

    @Test
    void testInvalidApiEnvironment() {
        Properties invalidEnvProps = new Properties();
        invalidEnvProps.setProperty("api.environment", "staging"); // Invalid value
        // Add other mandatory fields
        invalidEnvProps.setProperty("api.userID", "dummy");
        invalidEnvProps.setProperty("api.password", "dummy");
        invalidEnvProps.setProperty("api.panOrDOB", "dummy");
        invalidEnvProps.setProperty("api.vendorId", "dummy");
        invalidEnvProps.setProperty("api.apiKey", "dummy");
        invalidEnvProps.setProperty("api.sourceId", "dummy");
        invalidEnvProps.setProperty("app.browserName", "dummy");
        invalidEnvProps.setProperty("app.browserVersion", "dummy");
        invalidEnvProps.setProperty("api.baseUrl.uat", "https://test.uat.api");
        invalidEnvProps.setProperty("api.baseUrl.live", "https://test.live.api");


        String tempInvalidEnvFile = "temp_invalid_env_config.properties";
        try {
            Path tempPath = Paths.get("target/test-classes/" + tempInvalidEnvFile);
            Files.createDirectories(tempPath.getParent());
            try (OutputStream out = Files.newOutputStream(tempPath)) {
                invalidEnvProps.store(out, "Temporary invalid env config");
            }
            AppConfig.loadProperties(tempInvalidEnvFile, true);

            Exception exception = assertThrows(RuntimeException.class, () -> {
                AppConfig.getBaseUrl();
            });
            assertTrue(exception.getMessage().contains("Invalid value for 'api.environment'"));
            Files.deleteIfExists(tempPath);
        } catch (IOException e) {
            fail("Failed to create or delete temporary config file for testInvalidApiEnvironment: " + e.getMessage());
        }
    }
    
    @Test
    void testPropertiesFileNotFound_Critical() {
        // This test relies on loadProperties throwing an exception for a critical file.
        // The static block in AppConfig itself would call System.exit or throw, 
        // but here we are testing the behavior of the loadProperties method directly.
        
        String nonExistentFile = "non_existent_config.properties";
        
        AppConfigInitializationException exception = assertThrows(AppConfigInitializationException.class, () -> {
            AppConfig.loadProperties(nonExistentFile, true); // Critical load
        });
        assertTrue(exception.getMessage().contains("Configuration file '" + nonExistentFile + "' not found in classpath."));
    }

    @Test
    void testPropertiesFileNotFound_NonCritical() {
        // For a non-critical load, no exception should be thrown, but an error logged.
        // We can't easily check System.err output in standard JUnit without extensions.
        // So, we'll just verify no exception is thrown and properties might be empty or defaults.
        String nonExistentFile = "non_existent_config_non_critical.properties";
        
        assertDoesNotThrow(() -> {
            AppConfig.loadProperties(nonExistentFile, false); // Non-critical load
        });
        // After a failed non-critical load, properties should be empty (as loadProperties clears them)
        // or hold defaults if the AppConfig class had them.
        // Let's check if a known property from a previous successful load (if any) is gone.
        // Or, more robustly, check if a property that would be in default_config is not there.
        // For this test, we'll assert that a property expected from test_config is not found.
        // This requires ensuring that test_config was loaded before this test, or that properties are cleared.
        // The `loadProperties` clears, so we can check for a key that would be in a valid file.
        RuntimeException exception = assertThrows(RuntimeException.class, () -> {
            AppConfig.getUserId(); // Accessing a property should fail if the file wasn't loaded
        });
        assertTrue(exception.getMessage().contains("Missing required value for key 'api.userID'"));
    }
}

// Required for testGetBaseUrl_LIVE and testMissingRequiredProperty to write temp files.
// Consider adding this import if not already present, though Files.newOutputStream should work.
// import java.io.FileOutputStream;
import java.io.OutputStream;
