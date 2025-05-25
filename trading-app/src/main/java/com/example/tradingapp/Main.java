package com.example.tradingapp;

import org.json.JSONObject;
import java.util.Scanner;
import java.util.Arrays;
import java.util.List;
import java.text.SimpleDateFormat;
import java.text.ParseException;

/**
 * Main class for the Trading Application Command-Line Interface (CLI).
 * This class provides the entry point for the application and handles user interaction
 * for logging in and placing trading orders. It uses {@link ApiService} to communicate
 * with the trading API and {@link AppConfig} for configuration.
 */
public class Main {
    private static ApiService apiService;
    private static Scanner scanner; // Scanner for user input throughout the application

    /**
     * The main entry point for the application.
     * Initializes services, handles the login process, facilitates order placement,
     * and manages overall application flow.
     * 
     * @param args Command-line arguments (not currently used).
     */
    public static void main(String[] args) {
        System.out.println("Starting Trading App CLI...");
        scanner = new Scanner(System.in);
        try {
            // ApiService constructor will load AppConfig and initialize the MOFSL API client.
            // If AppConfig fails to load critical properties, it throws AppConfigInitializationException,
            // which is a RuntimeException and will be caught by the generic RuntimeException handler below.
            apiService = new ApiService();

            if (!handleLogin()) {
                System.out.println("Login sequence failed. Exiting application.");
                // No System.exit(1) here to allow finally block to run for cleanup.
                return; 
            }

            System.out.println("\nLogin successful!");

            // Proceed to order placement or other functionalities
            PlaceOrderinfoParams orderParams = getOrderDetailsFromUser();
            if (orderParams != null) { // User might theoretically cancel or input might be invalid
                JSONObject orderResponse = apiService.placeOrder(
                        orderParams.exchange, orderParams.symbolToken, orderParams.buyOrSell,
                        orderParams.orderType, orderParams.productType, orderParams.orderDuration,
                        orderParams.price, orderParams.triggerPrice, orderParams.qtyInLot,
                        orderParams.disclosedQty, orderParams.amoOrder, orderParams.goodTillDate,
                        orderParams.tag
                );
                handleOrderResponse(orderResponse); // Display formatted response
            }

        } catch (org.json.JSONException e) { 
            System.err.println("JSON PROCESSING ERROR in Main: " + e.getMessage());
            e.printStackTrace(System.err);
            System.out.println("A severe error occurred while processing data. Please check logs.");
        } catch (java.util.InputMismatchException e) { // If scanner.nextTYPE() was used and failed
            System.err.println("INPUT ERROR in Main: An invalid input type was provided.");
            e.printStackTrace(System.err);
            System.out.println("An error occurred due to invalid input. Please ensure inputs match expected types (e.g., numbers for numeric fields).");
        } catch (AppConfigInitializationException e) { // Catch specific config errors that AppConfig now throws
             System.err.println("APPLICATION CONFIGURATION ERROR: " + e.getMessage());
             // Stack trace already printed by AppConfig for fatal errors.
             System.out.println("Application cannot start due to a configuration problem. Please correct the configuration and try again.");
             // System.exit(1) might have already happened in AppConfig if it was truly fatal and unrecoverable.
        }
        catch (RuntimeException e) { // Catch other runtime exceptions
            System.err.println("APPLICATION RUNTIME ERROR in Main: " + e.getMessage());
            e.printStackTrace(System.err);
             System.out.println("A critical runtime error occurred. Please check error logs. Message: " + e.getMessage());
        } catch (Exception e) { // Catch-all for any other unexpected exceptions
            System.err.println("UNEXPECTED APPLICATION ERROR in Main: " + e.getMessage());
            e.printStackTrace(System.err);
            System.out.println("An unexpected error occurred. Please check error logs.");
        } finally {
            if (scanner != null) {
                System.out.println("\nClosing scanner...");
                scanner.close();
            }
            System.out.println("Exiting Trading App CLI.");
        }
    }

    /**
     * Handles the user login process, including OTP verification if required by the API.
     * 
     * @return true if login (and OTP verification, if applicable) is successful, false otherwise.
     */
    private static void handleOrderResponse(JSONObject orderResponse) {
        System.out.println("\n--- Place Order Response ---");
        if (orderResponse == null) {
            System.out.println("Received no response from the server (response was null).");
            System.out.println("--------------------------");
            return;
        }

        try {
            // First, check for the assumed common pattern: "stat"
            if (orderResponse.has("stat")) {
                String stat = orderResponse.getString("stat");
                if ("Ok".equalsIgnoreCase(stat)) {
                    System.out.println("Order placed successfully!");
                    // Try to find an order ID, checking common names
                    if (orderResponse.has("orderId")) {
                        System.out.println("Order ID: " + orderResponse.getString("orderId"));
                    } else if (orderResponse.has("uniqueorderid")) {
                        System.out.println("Unique Order ID: " + orderResponse.getString("uniqueorderid"));
                    } else if (orderResponse.has("BrokerOrderID")) { // From placeholder
                        System.out.println("Broker Order ID: " + orderResponse.getString("BrokerOrderID"));
                    } else {
                        System.out.println("Confirmation received, but Order ID not found in the response.");
                    }
                } else if ("Not_Ok".equalsIgnoreCase(stat)) {
                    System.out.println("Order placement failed.");
                    if (orderResponse.has("emsg")) {
                        System.out.println("Reason: " + orderResponse.getString("emsg"));
                    } else if (orderResponse.has("Message")) { // From placeholder
                        System.out.println("Reason (from Message field): " + orderResponse.getString("Message"));
                    } else {
                        System.out.println("Error status received, but no specific error message found.");
                    }
                } else {
                    // "stat" field has an unexpected value
                    System.out.println("Received an unexpected status value in 'stat': " + stat);
                    System.out.println("Raw Response: " + orderResponse.toString());
                }
            }
            // If "stat" is not present, check for placeholder's "Status" (integer)
            else if (orderResponse.has("Status")) {
                int statusInt = orderResponse.getInt("Status");
                if (statusInt == 0) { // Assuming 0 from placeholder means success
                    System.out.println("Order placed successfully! (Status code: 0)");
                    if (orderResponse.has("BrokerOrderID")) {
                        System.out.println("Broker Order ID: " + orderResponse.getString("BrokerOrderID"));
                    } else if (orderResponse.has("orderId")) {
                        System.out.println("Order ID: " + orderResponse.getString("orderId"));
                    } else if (orderResponse.has("uniqueorderid")) {
                        System.out.println("Unique Order ID: " + orderResponse.getString("uniqueorderid"));
                    } else {
                        System.out.println("Confirmation received, but Order ID not found in the response.");
                    }
                } else { // Non-zero "Status" from placeholder means error
                    System.out.println("Order placement failed. (Status code: " + statusInt + ")");
                    if (orderResponse.has("Message")) {
                        System.out.println("Reason: " + orderResponse.getString("Message"));
                    } else if (orderResponse.has("emsg")) {
                        System.out.println("Reason (from emsg field): " + orderResponse.getString("emsg"));
                    } else {
                        System.out.println("Error status code received, but no specific error message found.");
                    }
                }
            }
            // If neither "stat" nor "Status" is present
            else {
                System.out.println("Received an unexpected response structure from the server (missing 'stat' or 'Status' field).");
                System.out.println("Raw Response: " + orderResponse.toString());
            }
        } catch (org.json.JSONException e) {
            System.err.println("Error parsing the JSON response from order placement: " + e.getMessage());
            System.out.println("Raw Response (due to parsing error): " + orderResponse.toString());
            // e.printStackTrace(); // Optionally print stack trace for debugging
        }
        System.out.println("--------------------------");
    }

    private static boolean handleLogin() {
        System.out.println("\n--- Login ---");
        System.out.println("\n--- Login Process ---");
        JSONObject loginResponse = apiService.login(); // Attempt initial login

        // ApiService ensures loginResponse is not null and contains "Status"
            int status = loginResponse.getInt("Status"); 

        if (status == 2) { // Status 2 indicates OTP is required (based on placeholder SDK behavior)
            System.out.println("Login requires OTP. Server Message: " + loginResponse.optString("Message"));
            String otp;
            // Loop until a 6-digit numeric OTP is entered
            while (true) {
                System.out.print("Enter 6-digit OTP received on Mobile/Email: ");
                otp = scanner.nextLine().trim();
                if (otp.length() == 6 && otp.matches("\\d+")) {
                    break; // Valid OTP format
                }
                System.out.println("Invalid OTP format. OTP must be 6 digits numeric.");
            }

            JSONObject otpResponse = apiService.verifyOtpAndLogin(otp);
            // ApiService ensures otpResponse is not null and contains "Status"
                int otpStatus = otpResponse.getInt("Status");
            if (otpStatus == 0 && apiService.isLoggedIn()) {
                System.out.println("OTP Verified successfully. Login complete.");
                return true;
            } else {
                System.err.println("ERROR: OTP Verification failed. Server Message: " + otpResponse.optString("Message"));
                return false;
            }
        } else if (status == 0 && apiService.isLoggedIn()) {
            // Login successful without needing a separate OTP step (e.g., TOTP from config worked)
            System.out.println("Login successful (no separate OTP step required).");
                return true; 
        } else {
            // Login failed at the first step
            System.err.println("ERROR: Login failed. Server Message: " + loginResponse.optString("Message") + " (Status Code: " + status + ")");
            return false;
        }
    }

    /**
     * Internal class to hold parameters for placing an order, gathered from user input.
     */
    private static class PlaceOrderinfoParams {
        String exchange;
        int symbolToken;
        String buyOrSell;
        String orderType;
        String productType;
        String orderDuration;
        float price;
        int qtyInLot; // Changed from float to int as per PlaceOrderinfo.java
        int triggerPrice; // Changed from float to int
        int disclosedQty;
        String amoOrder;
        String goodTillDate;
        String tag;
    }

    /**
     * Prompts the user to enter details for placing a trading order.
     * Performs basic client-side validation for each input field.
     * 
     * @return A {@link PlaceOrderinfoParams} object populated with the user's input,
     *         or null if the user cancels or an irrecoverable input error occurs (not currently implemented).
     */
    private static PlaceOrderinfoParams getOrderDetailsFromUser() {
        System.out.println("\n--- Enter Order Details ---");
        PlaceOrderinfoParams params = new PlaceOrderinfoParams();

        // Exchange
        List<String> validExchanges = Arrays.asList("NSE", "BSE", "NFO", "MCX", "CDS"); 
        params.exchange = getValidatedStringInput("Exchange (" + String.join("/", validExchanges) + "): ", validExchanges, true);

        // Symbol Token
        params.symbolToken = getPositiveIntegerInput("Symbol Token (e.g., 22 for RELIANCE on NSE): ");

        // Buy or Sell
        List<String> validBuySell = Arrays.asList("BUY", "SELL");
        params.buyOrSell = getValidatedStringInput("Transaction Type (BUY/SELL): ", validBuySell, true);

        // Order Type
        List<String> validOrderTypes = Arrays.asList("LIMIT", "MARKET", "SL", "SL-M"); // SL = Stop Loss Limit, SL-M = Stop Loss Market
        params.orderType = getValidatedStringInput("Order Type (" + String.join("/", validOrderTypes) + "): ", validOrderTypes, true);
        
        // Product Type
        List<String> validProductTypes = Arrays.asList("NORMAL", "INTRADAY", "VALUEPLUS", "MARGIN", "CNC", "MTF"); 
        params.productType = getValidatedStringInput("Product Type (" + String.join("/", validProductTypes) + "): ", validProductTypes, true);

        // Order Duration
        List<String> validOrderDurations = Arrays.asList("DAY", "IOC", "GTC", "GTD"); 
        params.orderDuration = getValidatedStringInput("Order Duration (" + String.join("/", validOrderDurations) + "): ", validOrderDurations, true);
        
        // Price (Conditional based on Order Type)
        if (params.orderType.equals("MARKET") || params.orderType.equals("SL-M")) {
            params.price = 0; // Market orders use 0 for price.
            System.out.println("Info: Price set to 0 for " + params.orderType + " order.");
        } else { // LIMIT or SL orders require a price
            params.price = getPositiveFloatInput("Price (decimal, must be > 0 for " + params.orderType + "): ");
        }
        
        // Quantity (in lots)
        params.qtyInLot = getPositiveIntegerInput("Quantity (integer, in number of shares/lots): ");
        
        // Trigger Price (Conditional based on Order Type)
        if (params.orderType.equals("SL") || params.orderType.equals("SL-M")) {
            params.triggerPrice = (int) getPositiveFloatInput("Trigger Price (decimal, must be > 0 for SL/SL-M): ");
        } else {
            params.triggerPrice = 0; // Not applicable for LIMIT or MARKET
        }
        
        // Disclosed Quantity (optional)
        params.disclosedQty = getNonNegativeIntegerInput("Disclosed Quantity (integer, optional, default 0): ", true);

        // AMO Order (optional)
        List<String> validAmo = Arrays.asList("Y", "N");
        params.amoOrder = getValidatedStringInput("Is this an After Market Order (AMO)? (Y/N, default N): ", validAmo, false);
        if (params.amoOrder.isEmpty()) params.amoOrder = "N"; // Default to "N" if user just presses Enter
        
        // Good Till Date (optional, relevant for GTD/GTC, though GTC is often system managed)
        params.goodTillDate = getOptionalDateInput("Good Till Date (DD-MMM-YYYY, optional, e.g., 03-JUN-2024): ");
        if (params.orderDuration.equals("GTD") && params.goodTillDate.isEmpty()) {
            System.out.println("Warning: GTD orders typically require a Good Till Date. The order might be rejected or treated as DAY.");
        }
        if (!params.orderDuration.equals("GTD") && !params.goodTillDate.isEmpty()) {
             System.out.println("Info: Good Till Date is usually for GTD orders. It might be ignored for " + params.orderDuration + " orders.");
        }

        // Tag (optional)
        System.out.print("Order Tag (string, optional, max 20 chars): ");
        params.tag = scanner.nextLine().trim();
        if (params.tag.length() > 20) { // Example of a simple constraint
            System.out.println("Warning: Tag is longer than 20 characters, it might be truncated by the API.");
            params.tag = params.tag.substring(0, 20);
        }
        return params;
    }

    /**
     * Prompts the user for a string input and validates it against a list of allowed values.
     * Loops until valid input is provided.
     * 
     * @param prompt The message to display to the user.
     * @param validValues A list of valid string inputs.
     * @param toUpperCase If true, the user's input is converted to uppercase before validation.
     * @return The validated user input.
     */
    private static String getValidatedStringInput(String prompt, List<String> validValues, boolean toUpperCase) {
        String input;
        while (true) {
            System.out.print(prompt);
            input = scanner.nextLine().trim();
            if (toUpperCase) {
                input = input.toUpperCase();
            }
            if (validValues.contains(input)) {
                return input;
            } else if (input.isEmpty() && prompt.toLowerCase().contains("optional")) { 
                // More generic check for optional empty string if prompt indicates it
                if (prompt.contains("AMO Order") && validValues.contains("N")) return "N"; // specific default for AMO
                return ""; // General optional empty
            }
            System.out.println("Invalid input. Expected one of: " + String.join(", ", validValues) + ". Please try again.");
        }
    }
    
    /**
     * Prompts the user for an optional date string and validates its format (DD-MMM-YYYY).
     * Loops until a valid format is entered or the input is left empty.
     * 
     * @param prompt The message to display to the user.
     * @return The validated date string, or an empty string if the user provides no input.
     */
    private static String getOptionalDateInput(String prompt) {
        SimpleDateFormat sdf = new SimpleDateFormat("dd-MMM-yyyy");
        sdf.setLenient(false); // Strict date parsing
        String input;
        while (true) {
            System.out.print(prompt);
            input = scanner.nextLine().trim();
            if (input.isEmpty()) {
                return ""; // Optional, so empty is fine
            }
            try {
                sdf.parse(input); // Check if the date is in the correct format
                return input; 
            } catch (ParseException e) {
                System.out.println("Invalid date format. Please use DD-MMM-YYYY (e.g., 03-JUN-2024) or leave empty.");
            }
        }
    }

    /**
     * Prompts the user for an integer input that must be positive (greater than 0).
     * Loops until valid input is provided.
     * 
     * @param prompt The message to display to the user.
     * @return The validated positive integer.
     */
    private static int getPositiveIntegerInput(String prompt) {
        int value;
        while (true) {
            System.out.print(prompt);
            String line = scanner.nextLine().trim();
            try {
                value = Integer.parseInt(line);
                if (value > 0) {
                    return value;
                } else {
                    System.out.println("Invalid input: Please enter a positive whole number greater than 0.");
                }
            } catch (NumberFormatException e) {
                System.out.println("Invalid input: Not a valid whole number. Please try again.");
            }
        }
    }
    
    /**
     * Prompts the user for an integer input that must be non-negative (0 or more).
     * Can be configured to be optional, returning a default value (0) if empty.
     * Loops until valid input is provided.
     * 
     * @param prompt The message to display to the user.
     * @param optional If true and the user enters nothing, returns 0.
     * @return The validated non-negative integer.
     */
    private static int getNonNegativeIntegerInput(String prompt, boolean optional) {
        int value;
        while (true) {
            System.out.print(prompt);
            String line = scanner.nextLine().trim();
            if (optional && line.isEmpty()) {
                return 0; // Default for optional non-negative integer
            }
            try {
                value = Integer.parseInt(line);
                if (value >= 0) {
                    return value;
                } else {
                    System.out.println("Invalid input: Please enter a non-negative whole number (0 or greater).");
                }
            } catch (NumberFormatException e) {
                System.out.println("Invalid input: Not a valid whole number. Please try again.");
            }
        }
    }

    /**
     * Prompts the user for a float input that must be non-negative (0.0 or more).
     * Loops until valid input is provided.
     * 
     * @param prompt The message to display to the user.
     * @return The validated non-negative float.
     */
    private static float getNonNegativeFloatInput(String prompt) {
        float value;
        while (true) {
            System.out.print(prompt);
            String line = scanner.nextLine().trim();
            try {
                value = Float.parseFloat(line);
                if (value >= 0) {
                    return value;
                } else {
                    System.out.println("Invalid input: Please enter a non-negative decimal number (0 or more).");
                }
            } catch (NumberFormatException e) {
                System.out.println("Invalid input: Not a valid decimal number. Please try again.");
            }
        }
    }
}
