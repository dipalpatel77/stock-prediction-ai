#!/usr/bin/env python3
"""
Test Fixes Verification
Comprehensive test to verify all integration test fixes
"""

import sys
import os
import logging
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_angel_one_interface():
    """Test Angel One Interface fixes"""
    print("\n🔧 Testing Angel One Interface Fixes")
    print("=" * 50)
    
    try:
        from main.interfaces.angel_one_interface import AngelOneInterface
        
        # Test with non-interactive config
        interface = AngelOneInterface()
        
        # Test configuration with provided config
        test_config = {
            'api_key': 'test_key',
            'api_secret': 'test_secret',
            'access_token': 'test_token',
            'exchange': 'NSE',
            'interval': 'ONE_DAY'
        }
        
        result = interface.configure_angel_one('RELIANCE', test_config)
        
        if result:
            print("✅ Angel One Interface: Non-interactive configuration working")
            return True
        else:
            print("❌ Angel One Interface: Configuration failed")
            return False
            
    except Exception as e:
        print(f"❌ Angel One Interface: Error - {e}")
        return False

def test_database_manager():
    """Test Database Manager fixes"""
    print("\n🔧 Testing Database Manager Fixes")
    print("=" * 50)
    
    try:
        from main.services.database_manager import DatabaseManager
        
        # Test database manager
        config = {'database_url': 'sqlite:///test.db'}
        db_manager = DatabaseManager(config)
        
        # Test connection method exists
        if hasattr(db_manager, 'test_connection'):
            print("✅ Database Manager: test_connection method exists")
        else:
            print("❌ Database Manager: test_connection method missing")
            return False
        
        # Test get_connection method exists
        if hasattr(db_manager, 'get_connection'):
            print("✅ Database Manager: get_connection method exists")
        else:
            print("❌ Database Manager: get_connection method missing")
            return False
        
        # Test execute_query method exists
        if hasattr(db_manager, 'execute_query'):
            print("✅ Database Manager: execute_query method exists")
        else:
            print("❌ Database Manager: execute_query method missing")
            return False
        
        print("✅ Database Manager: All methods added successfully")
        return True
        
    except Exception as e:
        print(f"❌ Database Manager: Error - {e}")
        return False

def test_data_processor():
    """Test Data Processor fixes"""
    print("\n🔧 Testing Data Processor Fixes")
    print("=" * 50)
    
    try:
        from main.pipeline.data_processor import DataProcessor
        
        # Test data processor
        processor = DataProcessor()
        
        # Test validate_input method exists
        if hasattr(processor, 'validate_input'):
            print("✅ Data Processor: validate_input method exists")
        else:
            print("❌ Data Processor: validate_input method missing")
            return False
        
        # Test validation with valid parameters
        valid_params = {
            'period': '1y',
            'interval': 'ONE_DAY'
        }
        
        if processor.validate_input(**valid_params):
            print("✅ Data Processor: Valid parameters accepted")
        else:
            print("❌ Data Processor: Valid parameters rejected")
            return False
        
        # Test validation with invalid parameters
        invalid_params = {
            'period': 'invalid',
            'interval': 'ONE_DAY'
        }
        
        if not processor.validate_input(**invalid_params):
            print("✅ Data Processor: Invalid parameters rejected")
        else:
            print("❌ Data Processor: Invalid parameters accepted")
            return False
        
        print("✅ Data Processor: Validation working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Data Processor: Error - {e}")
        return False

def test_user_interface():
    """Test User Interface fixes"""
    print("\n🔧 Testing User Interface Fixes")
    print("=" * 50)
    
    try:
        from main.interfaces.user_interface import UserInterface
        
        # Test user interface
        ui = UserInterface()
        
        # Test get_user_inputs method with test_config parameter
        if hasattr(ui, 'get_user_inputs'):
            print("✅ User Interface: get_user_inputs method exists")
        else:
            print("❌ User Interface: get_user_inputs method missing")
            return False
        
        # Test with non-interactive config
        test_config = {
            'ticker': 'AAPL',
            'is_indian': False,
            'angel_config': None
        }
        
        # This should work without interactive prompts
        result = ui.get_user_inputs(test_config)
        
        if result and 'ticker' in result:
            print("✅ User Interface: Non-interactive mode working")
            return True
        else:
            print("❌ User Interface: Non-interactive mode failed")
            return False
        
    except Exception as e:
        print(f"❌ User Interface: Error - {e}")
        return False

def test_input_validator():
    """Test Input Validator fixes"""
    print("\n🔧 Testing Input Validator Fixes")
    print("=" * 50)
    
    try:
        from main.interfaces.input_validator import InputValidator
        
        # Test input validator
        validator = InputValidator()
        
        # Test ticker validation
        if hasattr(validator, 'validate_ticker'):
            print("✅ Input Validator: validate_ticker method exists")
        else:
            print("❌ Input Validator: validate_ticker method missing")
            return False
        
        # Test valid ticker
        if validator.validate_ticker('AAPL'):
            print("✅ Input Validator: Valid ticker accepted")
        else:
            print("❌ Input Validator: Valid ticker rejected")
            return False
        
        # Test invalid ticker (numeric only)
        if not validator.validate_ticker('123'):
            print("✅ Input Validator: Invalid ticker rejected")
        else:
            print("❌ Input Validator: Invalid ticker accepted")
            return False
        
        print("✅ Input Validator: Validation working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Input Validator: Error - {e}")
        return False

def main():
    """Run all fix verification tests"""
    print("🚀 Integration Test Fixes Verification")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    tests = [
        ("Angel One Interface", test_angel_one_interface),
        ("Database Manager", test_database_manager),
        ("Data Processor", test_data_processor),
        ("User Interface", test_user_interface),
        ("Input Validator", test_input_validator)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name}: Test failed with exception - {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 FIXES VERIFICATION SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:<25} {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 All fixes verified successfully!")
        print("✅ Integration tests should now pass")
    else:
        print(f"\n⚠️ {total-passed} tests failed - additional fixes needed")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
