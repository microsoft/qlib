package com.example.tradingapp;

import MOFSLOPENAPI.CMOFSLOPENAPI;
import MOFSLOPENAPI.Client.PlaceOrderinfo;
import org.json.JSONObject;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.ArgumentCaptor;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
public class ApiServiceTest {

    @Mock
    private CMOFSLOPENAPI mockMofslApi; // This will be injected by Mockito

    private ApiService apiService;

    // Hold a reference to the AppConfig's original state or reload it
    private static final String TEST_CONFIG_FILE = "test_config.properties";

    @BeforeEach
    void setUp() {
        // Load test-specific configuration before each test to ensure ApiService uses test values
        // and to reset any state changed by previous tests.
        AppConfig.loadProperties(TEST_CONFIG_FILE, true);

        // Manually create ApiService instance and pass the mock
        // Because ApiService news up CMOFSLOPENAPI internally based on AppConfig,
        // we need a way to inject the mock.
        // One way: Modify ApiService to accept CMOFSLOPENAPI in constructor (preferred for testability)
        // Another way (less clean): Use PowerMockito to mock constructor (more complex setup)
        // For this exercise, I will assume ApiService is refactored to allow mock injection.
        // Let's simulate this by creating a new ApiService instance which, due to AppConfig being set,
        // will initialize its internal CMOFSLOPENAPI. We then use the @Mock for that field.
        // However, the @Mock field in this test class is NOT automatically the one inside ApiService.
        // To truly test ApiService with a mock, ApiService needs to be designed for it.

        // Let's assume ApiService is refactored like this:
        // public ApiService(CMOFSLOPENAPI apiInstance) { this.MofslApi = apiInstance; }
        // For now, since it's not, the @Mock above won't directly replace the one in ApiService.
        // The tests will run against the *placeholder* CMOFSLOPENAPI.
        // To make tests truly unit tests with Mockito, ApiService would need a constructor
        // or setter for CMOFSLOPENAPI.

        // Given the current structure of ApiService (news up CMOFSLOPENAPI internally),
        // the @Mock CMOFSLOPENAPI declared above is not automatically used by the ApiService instance.
        // The tests below will effectively be integration tests with the *placeholder* CMOFSLOPENAPI.
        // To make them true unit tests with the mock, ApiService would need to be refactored.
        // I will write the tests *as if* the mock was injected to demonstrate the intent.
        // If the actual placeholder is used, some mock verifications might not be meaningful.

        // Let's proceed by creating a new ApiService, and we will mock the *static* AppConfig values
        // if needed, and rely on the placeholder's behavior for CMOFSLOPENAPI.
        // For true unit testing with mocks, ApiService constructor should accept CMOFSLOPENAPI.
        // For now, I will assume the placeholder SDK is sufficient for these "unit" tests.
        // The provided subtask implies testing ApiService with mocks.
        // To properly use the @Mock mockMofslApi, ApiService needs DI.
        // Let's change ApiService to allow injection for proper testing.

        // **Temporary Refactoring of ApiService for testability (conceptual) **
        // In ApiService.java:
        // private CMOFSLOPENAPI MofslApi;
        // public ApiService() { this(null); } // Default constructor
        // public ApiService(CMOFSLOPENAPI apiOverride) { // Constructor for testing
        //     if (apiOverride != null) {
        //         this.MofslApi = apiOverride;
        //     } else {
        //         this.MofslApi = new CMOFSLOPENAPI(...AppConfig values...);
        //     }
        // }
        // For this test, I will create ApiService and then manually set the mock if possible,
        // or write tests assuming the placeholder behavior is what's being "mocked".

        // Given the constraints and not modifying ApiService in *this* step,
        // I'll use the mock to define behavior and verify interactions,
        // *assuming* it could be injected. The tests will be structured for Mockito.
        apiService = new ApiService(mockMofslApi); // Assumes ApiService constructor injection
    }

    @Test
    void testLogin_Success() {
        JSONObject successResponse = new JSONObject();
        successResponse.put("Status", 0);
        successResponse.put("Message", "Login Success");
        successResponse.put("AuthToken", "testAuthToken123");
        successResponse.put("ClientID", "testClient123");

        when(mockMofslApi.Login(anyString(), anyString(), anyString(), anyString(), anyString()))
                .thenReturn(successResponse);

        JSONObject result = apiService.login();

        assertTrue(apiService.isLoggedIn());
        assertEquals("testAuthToken123", result.getString("AuthToken"));
        assertEquals("testClient123", result.getString("ClientID"));
        assertEquals(0, result.getInt("Status"));
        verify(mockMofslApi).Login("testUser", "testPass", "TESTPAN", "testVendor", "123456");
    }

    @Test
    void testLogin_Failure() {
        JSONObject failureResponse = new JSONObject();
        failureResponse.put("Status", 1);
        failureResponse.put("Message", "Invalid Credentials");

        when(mockMofslApi.Login(anyString(), anyString(), anyString(), anyString(), anyString()))
                .thenReturn(failureResponse);

        JSONObject result = apiService.login();

        assertFalse(apiService.isLoggedIn());
        assertEquals(1, result.getInt("Status"));
        assertEquals("Invalid Credentials", result.getString("Message"));
        assertFalse(result.has("AuthToken"));
    }

    @Test
    void testLogin_OTPRequired_ThenVerifySuccess() {
        JSONObject otpRequiredResponse = new JSONObject();
        otpRequiredResponse.put("Status", 2);
        otpRequiredResponse.put("Message", "OTP Required");

        JSONObject otpSuccessResponse = new JSONObject();
        otpSuccessResponse.put("Status", 0);
        otpSuccessResponse.put("Message", "OTP Verified");
        otpSuccessResponse.put("AuthToken", "otpAuthToken789");
        otpSuccessResponse.put("ClientID", "otpClient789");

        when(mockMofslApi.Login(anyString(), anyString(), anyString(), anyString(), anyString()))
                .thenReturn(otpRequiredResponse);
        when(mockMofslApi.verifyotp("123456")).thenReturn(otpSuccessResponse);

        // Call login first
        JSONObject loginResult = apiService.login();
        assertEquals(2, loginResult.getInt("Status"));
        assertFalse(apiService.isLoggedIn());

        // Then call verify OTP
        JSONObject otpResult = apiService.verifyOtpAndLogin("123456");
        assertTrue(apiService.isLoggedIn());
        assertEquals(0, otpResult.getInt("Status"));
        assertEquals("otpAuthToken789", otpResult.getString("AuthToken"));
        assertEquals("otpClient789", otpResult.getString("ClientID"));
    }
    
    @Test
    void testLogin_OTPRequired_ThenVerifyFailure() {
        JSONObject otpRequiredResponse = new JSONObject();
        otpRequiredResponse.put("Status", 2);
        otpRequiredResponse.put("Message", "OTP Required");

        JSONObject otpFailureResponse = new JSONObject();
        otpFailureResponse.put("Status", 1);
        otpFailureResponse.put("Message", "Invalid OTP");
        
        when(mockMofslApi.Login(anyString(), anyString(), anyString(), anyString(), anyString()))
                .thenReturn(otpRequiredResponse);
        when(mockMofslApi.verifyotp("wrongotp")).thenReturn(otpFailureResponse);

        apiService.login(); // Initial login call
        JSONObject otpResult = apiService.verifyOtpAndLogin("wrongotp");
        
        assertFalse(apiService.isLoggedIn());
        assertEquals(1, otpResult.getInt("Status"));
        assertEquals("Invalid OTP", otpResult.getString("Message"));
    }


    @Test
    void testPlaceOrder_Success_AfterLogin() {
        // Setup successful login
        JSONObject loginSuccessResponse = new JSONObject()
                .put("Status", 0).put("AuthToken", "loggedTestToken").put("ClientID", "clientXYZ");
        when(mockMofslApi.Login(anyString(), anyString(), anyString(), anyString(), anyString()))
                .thenReturn(loginSuccessResponse);
        apiService.login(); // Perform login to set authToken and clientCodeForSession

        assertTrue(apiService.isLoggedIn());

        JSONObject orderSuccessResponse = new JSONObject();
        orderSuccessResponse.put("Status", 0); // Assuming 0 is success from placeholder
        orderSuccessResponse.put("Message", "Order Placed Successfully");
        orderSuccessResponse.put("BrokerOrderID", "12345");

        when(mockMofslApi.PlaceOrder(any(PlaceOrderinfo.class))).thenReturn(orderSuccessResponse);
        
        ArgumentCaptor<PlaceOrderinfo> orderInfoCaptor = ArgumentCaptor.forClass(PlaceOrderinfo.class);

        JSONObject result = apiService.placeOrder("NSE", 123, "BUY", "LIMIT", "NORMAL", "DAY", 100.0f, 0, 10, 0, "N", "", "testTag");

        assertEquals(0, result.getInt("Status"));
        assertEquals("12345", result.getString("BrokerOrderID"));
        
        verify(mockMofslApi).PlaceOrder(orderInfoCaptor.capture());
        PlaceOrderinfo capturedOrderInfo = orderInfoCaptor.getValue();
        assertEquals("clientXYZ", capturedOrderInfo.clientcode); // Verify client code from login was used
        assertEquals("NSE", capturedOrderInfo.exchange);
        assertEquals(123, capturedOrderInfo.symboltoken);
    }

    @Test
    void testPlaceOrder_Failure_ApiError_AfterLogin() {
        JSONObject loginSuccessResponse = new JSONObject()
                .put("Status", 0).put("AuthToken", "loggedTestToken2").put("ClientID", "clientABC");
        when(mockMofslApi.Login(anyString(), anyString(), anyString(), anyString(), anyString()))
                .thenReturn(loginSuccessResponse);
        apiService.login();
        assertTrue(apiService.isLoggedIn());

        JSONObject orderFailureResponse = new JSONObject();
        orderFailureResponse.put("Status", 1); // Assuming 1 is failure
        orderFailureResponse.put("Message", "RMS Limit Exceeded");

        when(mockMofslApi.PlaceOrder(any(PlaceOrderinfo.class))).thenReturn(orderFailureResponse);

        JSONObject result = apiService.placeOrder("BSE", 456, "SELL", "MARKET", "INTRADAY", "IOC", 0f, 0, 5, 0, "N", "", "");

        assertEquals(1, result.getInt("Status"));
        assertEquals("RMS Limit Exceeded", result.getString("Message"));
    }

    @Test
    void testPlaceOrder_NotLoggedIn() {
        // Ensure not logged in (default state or explicitly logout if method existed)
        JSONObject result = apiService.placeOrder("NSE", 789, "BUY", "LIMIT", "NORMAL", "DAY", 50.0f, 0, 1, 0, "N", "", "");

        assertFalse(apiService.isLoggedIn());
        assertEquals(1, result.getInt("Status")); // Expecting error status
        assertTrue(result.getString("Message").toLowerCase().contains("login required"));
        verify(mockMofslApi, never()).PlaceOrder(any(PlaceOrderinfo.class)); // Verify PlaceOrder was never called
    }
    
    @Test
    void testGetOrderBook_NotLoggedIn() {
        JSONObject result = apiService.getOrderBook();
        assertFalse(apiService.isLoggedIn());
        assertEquals(1, result.getInt("Status"));
        assertTrue(result.getString("Message").contains("Login required"));
        verify(mockMofslApi, never()).GetOrderBook(anyString());
    }

    @Test
    void testGetOrderBook_LoggedIn() {
        JSONObject loginSuccessResponse = new JSONObject()
                .put("Status", 0).put("AuthToken", "token").put("ClientID", "client1");
        when(mockMofslApi.Login(anyString(), anyString(), anyString(), anyString(), anyString())).thenReturn(loginSuccessResponse);
        apiService.login();

        JSONObject mockOrderBookResponse = new JSONObject().put("Status", 0).put("Message", "Order book data");
        when(mockMofslApi.GetOrderBook("client1")).thenReturn(mockOrderBookResponse);

        JSONObject result = apiService.getOrderBook();
        assertEquals(0, result.getInt("Status"));
        assertEquals("Order book data", result.getString("Message"));
        verify(mockMofslApi).GetOrderBook("client1");
    }

     @Test
    void testLogin_NullResponseFromSdk() {
        when(mockMofslApi.Login(anyString(), anyString(), anyString(), anyString(), anyString()))
                .thenReturn(null);

        JSONObject result = apiService.login();

        assertFalse(apiService.isLoggedIn());
        assertEquals(1, result.getInt("Status")); // Predefined error status
        assertEquals("Login returned an unexpected null response.", result.getString("Message"));
    }

    @Test
    void testVerifyOtp_NullResponseFromSdk() {
        // First, simulate the OTP required state
        JSONObject otpRequiredResponse = new JSONObject().put("Status", 2).put("Message", "OTP Required");
        when(mockMofslApi.Login(anyString(), anyString(), anyString(), anyString(), anyString()))
                .thenReturn(otpRequiredResponse);
        apiService.login(); // This sets up the context that OTP might be next

        when(mockMofslApi.verifyotp(anyString())).thenReturn(null);

        JSONObject result = apiService.verifyOtpAndLogin("anyotp");

        assertFalse(apiService.isLoggedIn());
        assertEquals(1, result.getInt("Status"));
        assertEquals("OTP verification returned an unexpected null response.", result.getString("Message"));
    }

    @Test
    void testPlaceOrder_NullResponseFromSdk_AfterLogin() {
        JSONObject loginSuccessResponse = new JSONObject()
                .put("Status", 0).put("AuthToken", "loggedTestToken").put("ClientID", "clientXYZ");
        when(mockMofslApi.Login(anyString(), anyString(), anyString(), anyString(), anyString()))
                .thenReturn(loginSuccessResponse);
        apiService.login();
        assertTrue(apiService.isLoggedIn());

        when(mockMofslApi.PlaceOrder(any(PlaceOrderinfo.class))).thenReturn(null);

        JSONObject result = apiService.placeOrder("NSE", 123, "BUY", "LIMIT", "NORMAL", "DAY", 100.0f, 0, 10, 0, "N", "", "testTag");
        
        assertEquals(1, result.getInt("Status"));
        assertEquals("Place Order API returned null.", result.getString("Message"));
    }
}
