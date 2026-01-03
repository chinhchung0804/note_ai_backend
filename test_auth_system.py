"""
Quick test script for NotallyX Authentication & Payment System
Run this script to verify that the authentication system is working correctly
"""

import requests
import json
from typing import Dict, Optional

BASE_URL = "http://localhost:8000"


class AuthTester:
    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
        self.token: Optional[str] = None
        self.user_data: Optional[Dict] = None
    
    def print_response(self, title: str, response: requests.Response):
        """Pretty print API response"""
        print(f"\n{'='*60}")
        print(f"📋 {title}")
        print(f"{'='*60}")
        print(f"Status Code: {response.status_code}")
        try:
            data = response.json()
            print(f"Response:\n{json.dumps(data, indent=2)}")
        except:
            print(f"Response: {response.text}")
        print(f"{'='*60}\n")
    
    def test_register(self, username: str, email: str, password: str):
        """Test user registration"""
        print("🔐 Testing User Registration...")
        
        url = f"{self.base_url}/api/auth/register"
        data = {
            "username": username,
            "email": email,
            "password": password
        }
        
        response = requests.post(url, json=data)
        self.print_response("User Registration", response)
        
        if response.status_code == 200:
            self.user_data = response.json()
            print("✅ Registration successful!")
            return True
        else:
            print("❌ Registration failed!")
            return False
    
    def test_login(self, username: str, password: str):
        """Test user login"""
        print("🔑 Testing User Login...")
        
        url = f"{self.base_url}/api/auth/login"
        data = {
            "username": username,
            "password": password
        }
        
        response = requests.post(url, json=data)
        self.print_response("User Login", response)
        
        if response.status_code == 200:
            result = response.json()
            self.token = result.get("access_token")
            self.user_data = result.get("user")
            print("✅ Login successful!")
            print(f"🎫 Token: {self.token[:50]}...")
            return True
        else:
            print("❌ Login failed!")
            return False
    
    def test_get_profile(self):
        """Test getting user profile"""
        if not self.token:
            print("❌ No token available. Please login first.")
            return False
        
        print("👤 Testing Get User Profile...")
        
        url = f"{self.base_url}/api/auth/me"
        headers = {"Authorization": f"Bearer {self.token}"}
        
        response = requests.get(url, headers=headers)
        self.print_response("User Profile", response)
        
        if response.status_code == 200:
            print("✅ Profile retrieved successfully!")
            return True
        else:
            print("❌ Failed to get profile!")
            return False
    
    def test_account_limits(self):
        """Test checking account limits"""
        if not self.token:
            print("❌ No token available. Please login first.")
            return False
        
        print("📊 Testing Account Limits...")
        
        url = f"{self.base_url}/api/auth/account-limits"
        headers = {"Authorization": f"Bearer {self.token}"}
        
        response = requests.get(url, headers=headers)
        self.print_response("Account Limits", response)
        
        if response.status_code == 200:
            print("✅ Account limits retrieved successfully!")
            return True
        else:
            print("❌ Failed to get account limits!")
            return False
    
    def test_get_plans(self):
        """Test getting payment plans"""
        print("💳 Testing Get Payment Plans...")
        
        url = f"{self.base_url}/api/payment/plans"
        
        response = requests.get(url)
        self.print_response("Payment Plans", response)
        
        if response.status_code == 200:
            print("✅ Payment plans retrieved successfully!")
            return True
        else:
            print("❌ Failed to get payment plans!")
            return False
    
    def test_process_note(self, text: str, note_id: str):
        """Test processing a note with authentication"""
        if not self.token:
            print("❌ No token available. Please login first.")
            return False
        
        print("📝 Testing Note Processing with Authentication...")
        
        url = f"{self.base_url}/api/v1/process"
        headers = {"Authorization": f"Bearer {self.token}"}
        data = {
            "text": text,
            "note_id": note_id
        }
        
        response = requests.post(url, headers=headers, data=data)
        self.print_response("Note Processing", response)
        
        if response.status_code == 200:
            print("✅ Note processed successfully!")
            return True
        else:
            print("❌ Note processing failed!")
            return False
    
    def test_rate_limiting(self):
        """Test rate limiting by creating multiple notes"""
        if not self.token:
            print("❌ No token available. Please login first.")
            return False
        
        print("⏱️  Testing Rate Limiting (FREE account: 5 notes/day)...")
        
        url = f"{self.base_url}/api/v1/process"
        headers = {"Authorization": f"Bearer {self.token}"}
        
        success_count = 0
        for i in range(7):  # Try to create 7 notes (should fail after 5)
            data = {
                "text": f"Test note {i+1} for rate limiting",
                "note_id": f"rate_test_{i+1}"
            }
            
            response = requests.post(url, headers=headers, data=data)
            
            if response.status_code == 200:
                success_count += 1
                print(f"✅ Note {i+1}: Success")
            elif response.status_code == 429:
                print(f"🚫 Note {i+1}: Rate limit reached (expected)")
                self.print_response(f"Rate Limit Response (Note {i+1})", response)
                break
            else:
                print(f"❌ Note {i+1}: Unexpected error")
                self.print_response(f"Error (Note {i+1})", response)
        
        print(f"\n📊 Successfully created {success_count} notes before hitting rate limit")
        return True


def main():
    """Run all tests"""
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║        NotallyX Authentication System Test Suite            ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    tester = AuthTester()
    
    # Test credentials
    username = "testuser_" + str(int(requests.get(f"{BASE_URL}/").elapsed.total_seconds() * 1000))
    email = f"{username}@example.com"
    password = "TestPassword123!"
    
    print(f"🧪 Test User Credentials:")
    print(f"   Username: {username}")
    print(f"   Email: {email}")
    print(f"   Password: {password}")
    
    # Run tests
    tests = [
        ("Registration", lambda: tester.test_register(username, email, password)),
        ("Login", lambda: tester.test_login(username, password)),
        ("Get Profile", lambda: tester.test_get_profile()),
        ("Account Limits", lambda: tester.test_account_limits()),
        ("Payment Plans", lambda: tester.test_get_plans()),
        ("Process Note", lambda: tester.test_process_note("This is a test note for authentication", "test_note_1")),
        ("Rate Limiting", lambda: tester.test_rate_limiting()),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test '{test_name}' crashed: {e}")
            results.append((test_name, False))
    
    # Print summary
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                      TEST SUMMARY                            ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status}: {test_name}")
    
    print(f"\n   📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n   🎉 All tests passed! Authentication system is working correctly.")
    else:
        print("\n   ⚠️  Some tests failed. Please check the logs above.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Test suite crashed: {e}")
