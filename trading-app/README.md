# Basic Trading CLI

## Overview

This application is a command-line interface (CLI) tool that connects to the Motilal Oswal Financial Services Ltd (MOFSL) Open API. It allows users to perform basic trading operations, specifically:
- Logging into their trading account (supports OTP verification if required by the API).
- Placing trading orders.

The application is built using Java and Maven.

## Prerequisites

Before you can build and run this application, please ensure you have the following:

1.  **Java 17 or higher:** The application is compiled using Java 17. You can check your Java version with `java -version`.
2.  **Maven:** Apache Maven is used for building the project. You can check your Maven version with `mvn -version`.
3.  **`MOFSLOPENAPI_V3.1.jar`:** This is the proprietary MOFSL Open API client library.
    *   **This JAR file is not available in any public Maven repository.** You must obtain this JAR file directly from MOFSL or your provider.
    *   Once obtained, you **must** place this `MOFSLOPENAPI_V3.1.jar` file into the `lib` directory located at the root of this project (`trading-app/lib/`). The project is configured to look for it there as a system dependency.
    *   A placeholder text file (`PUT_MOFSLOPENAPI_JAR_HERE.txt`) exists in the `lib` directory as a reminder.

## Configuration

All API credentials and necessary application settings must be configured in the `src/main/resources/config.properties` file. Before running the application, copy or rename the provided template/placeholder file (if one exists) or create this file and populate it with your actual details.

Key properties in `config.properties`:

*   `api.userID`: Your MOFSL User ID.
*   `api.password`: Your MOFSL account password.
*   `api.panOrDOB`: Your PAN number or Date of Birth (as required by the API for login).
*   `api.vendorId`: Your MOFSL assigned Vendor ID.
*   `api.totp`: Your Time-based One-Time Password (TOTP) if you use 2FA via an authenticator app. Leave blank if OTP is sent via SMS/Email or if not applicable.
*   `api.apiKey`: Your MOFSL API Key.
*   `api.clientCode`: Your client code (often obtained after the first login if not the same as UserID). Can be left blank initially if the API provides it after login.
*   `api.baseUrl.uat`: The base URL for the MOFSL UAT (testing) environment.
*   `api.baseUrl.live`: The base URL for the MOFSL LIVE (production) environment.
*   `api.environment`: Set this to either `uat` or `live` to determine which base URL the application will use.
*   `api.sourceId`: Source ID for API requests (e.g., "WEB").
*   `app.browserName`: Simulated browser name for API headers.
*   `app.browserVersion`: Simulated browser version for API headers.

**Important:** Ensure this file contains your valid credentials before attempting to run the application.

## How to Build

To build the application and package it into an executable JAR file:

1.  Navigate to the root directory of the project (i.e., the `trading-app` directory).
2.  Run the following Maven command:

    ```bash
    mvn clean package
    ```

3.  This command will compile the code, run tests (if any are not skipped), and create a JAR file in the `target/` directory (e.g., `trading-app-1.0-SNAPSHOT.jar`).

## How to Run

After successfully building the application:

1.  Ensure your `config.properties` file (`src/main/resources/config.properties`) is correctly filled out.
2.  Ensure the `MOFSLOPENAPI_V3.1.jar` is present in the `trading-app/lib/` directory.
3.  Open your terminal or command prompt, navigate to the `trading-app` directory.
4.  Run the application using the following command:

    ```bash
    java -jar target/trading-app-1.0-SNAPSHOT.jar
    ```
    (Note: The exact JAR filename might vary slightly based on the version in `pom.xml`.)

5.  The CLI will start, and you will be prompted for login (including OTP if the API indicates it's needed) and then for order details.

## Functionality

*   **Login:**
    *   Connects to the MOFSL API using credentials from `config.properties`.
    *   If the API requires OTP (e.g., if `api.totp` is not provided or is invalid), the user is prompted to enter the 6-digit OTP received on their registered mobile/email.
*   **Place Order:**
    *   After successful login, the user is prompted to enter various order details through a series of questions:
        *   Exchange (NSE, BSE, NFO, etc.)
        *   Symbol Token
        *   Transaction Type (BUY/SELL)
        *   Order Type (LIMIT, MARKET, SL, SL-M)
        *   Product Type (NORMAL, INTRADAY, etc.)
        *   Order Duration (DAY, IOC, GTD)
        *   Price (for LIMIT/SL orders)
        *   Quantity
        *   Trigger Price (for SL/SL-M orders)
        *   Optional fields: Disclosed Quantity, AMO status, Good Till Date, Order Tag.
    *   The application then sends the order request to the MOFSL API.
    *   The API's response (success or failure, with order ID or error message) is displayed to the user.

## Known Issues/Limitations

*   **Unit Test Coverage:**
    *   Unit tests for the input validation logic within `Main.java`'s helper methods are limited. These methods are private and tightly coupled with `java.util.Scanner`, making them difficult to unit test in isolation without significant refactoring.
*   **SDK Integration:**
    *   The `MOFSLOPENAPI_V3.1.jar` is treated as a black-box dependency. Its internal workings are not part of this project's source code.
    *   The application includes placeholder classes for `org.json.JSONObject`, `MOFSLOPENAPI.CMOFSLOPENAPI`, and `MOFSLOPENAPI.Client.PlaceOrderinfo`. These were necessary to allow the application to compile and demonstrate functionality, especially in an environment where the actual JAR might not be available or where testing with mocks is preferred. **When the real `MOFSLOPENAPI_V3.1.jar` is used, these placeholder classes should ideally be removed if the JAR provides them, and the project's dependencies should be configured to use the classes from the actual JAR.** (This project uses a system-scoped dependency for the JAR).
*   **Error Handling:** While basic error handling for API responses and input validation is present, it may not cover all edge cases or complex API error scenarios.
*   **Configuration Security:** API credentials are stored in plain text in `config.properties`. For production use, consider more secure ways to manage secrets.
*   **CLI Robustness:** The CLI is basic and may not handle all unexpected user inputs or terminal behaviors gracefully.
