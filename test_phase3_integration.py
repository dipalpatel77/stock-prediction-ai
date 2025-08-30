#!/usr/bin/env python3
"""
Phase 3 Integration Test Script
==============================

This script tests the Phase 3 integration including:
- Geopolitical risk assessment
- Corporate action tracking
- Insider trading analysis
- Combined analysis and insights
"""

import sys
import os
from datetime import datetime

def test_phase3_services():
    """Test individual Phase 3 services."""
    print("🧪 Testing Phase 3 Services...")
    print("=" * 50)
    
    try:
        # Test geopolitical risk service
        print("\n1. Testing Geopolitical Risk Service...")
        from core.geopolitical_risk_service import GeopoliticalRiskService
        
        geo_service = GeopoliticalRiskService()
        geo_risk = geo_service.assess_geopolitical_risk("AAPL")
        
        print(f"   ✅ Geopolitical Risk Score: {geo_risk.overall_risk_score:.1f}/100")
        print(f"   ✅ Event Count: {geo_risk.event_count}")
        print(f"   ✅ Risk Factors: {', '.join(geo_risk.risk_factors)}")
        print(f"   ✅ Market Sentiment Impact: {geo_risk.market_sentiment_impact:.1f}")
        
        # Test corporate action service
        print("\n2. Testing Corporate Action Service...")
        from core.corporate_action_service import CorporateActionService
        
        corp_service = CorporateActionService()
        corp_analysis = corp_service.analyze_corporate_actions("AAPL")
        
        print(f"   ✅ Total Actions: {corp_analysis.total_actions}")
        print(f"   ✅ Dividend Yield: {corp_analysis.dividend_yield:.1f}%")
        print(f"   ✅ Buyback Amount: ${corp_analysis.buyback_amount/1000000000:.1f}B")
        print(f"   ✅ Action Score: {corp_analysis.action_score:.1f}/100")
        
        # Test insider trading service
        print("\n3. Testing Insider Trading Service...")
        from core.insider_trading_service import InsiderTradingService
        
        insider_service = InsiderTradingService()
        insider_analysis = insider_service.analyze_insider_trading("AAPL")
        
        print(f"   ✅ Total Transactions: {insider_analysis.total_transactions}")
        print(f"   ✅ Insider Sentiment: {insider_analysis.insider_sentiment_score:.1f}/100")
        print(f"   ✅ Unusual Activity: {insider_analysis.unusual_activity_score:.1f}/100")
        print(f"   ✅ Net Activity: {insider_analysis.net_insider_activity:,} shares")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error testing services: {e}")
        return False

def test_phase3_integration():
    """Test Phase 3 integration."""
    print("\n🧪 Testing Phase 3 Integration...")
    print("=" * 50)
    
    try:
        from phase3_integration import Phase3Integration
        
        phase3 = Phase3Integration()
        analysis = phase3.run_phase3_analysis("AAPL")
        
        print(f"   ✅ Combined Risk Score: {analysis.combined_risk_score:.1f}/100")
        print(f"   ✅ Market Impact Score: {analysis.market_impact_score:.1f}/100")
        print(f"   ✅ Confidence Level: {analysis.confidence_level:.1f}/100")
        
        print(f"\n   📊 Key Insights ({len(analysis.key_insights)}):")
        for i, insight in enumerate(analysis.key_insights[:3], 1):
            print(f"      {i}. {insight}")
        
        print(f"\n   💡 Recommendations ({len(analysis.recommendations)}):")
        for i, rec in enumerate(analysis.recommendations[:3], 1):
            print(f"      {i}. {rec}")
        
        # Test summary
        summary = phase3.get_analysis_summary(analysis)
        print(f"\n   📋 Analysis Summary:")
        print(f"      Risk Level: {summary['risk_level']}")
        print(f"      Geopolitical Events: {summary['geopolitical_events']}")
        print(f"      Corporate Actions: {summary['corporate_actions']}")
        print(f"      Insider Transactions: {summary['insider_transactions']}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error testing integration: {e}")
        return False

def test_data_saving():
    """Test data saving functionality."""
    print("\n🧪 Testing Data Saving...")
    print("=" * 50)
    
    try:
        from phase3_integration import Phase3Integration
        
        phase3 = Phase3Integration()
        analysis = phase3.run_phase3_analysis("AAPL")
        
        # Test saving
        success = phase3.save_phase3_analysis("AAPL", analysis)
        
        if success:
            print("   ✅ Phase 3 analysis saved successfully")
            
            # Check if file exists
            import glob
            files = glob.glob("data/phase3/AAPL_phase3_analysis_*.json")
            if files:
                print(f"   ✅ Found {len(files)} saved analysis file(s)")
                latest_file = max(files, key=os.path.getctime)
                print(f"   ✅ Latest file: {os.path.basename(latest_file)}")
            else:
                print("   ⚠️ No saved files found")
        else:
            print("   ❌ Failed to save analysis")
        
        return success
        
    except Exception as e:
        print(f"   ❌ Error testing data saving: {e}")
        return False

def test_comprehensive_analysis():
    """Test comprehensive analysis with multiple tickers."""
    print("\n🧪 Testing Comprehensive Analysis...")
    print("=" * 50)
    
    tickers = ["AAPL", "MSFT", "GOOGL"]
    results = {}
    
    try:
        from phase3_integration import Phase3Integration
        
        phase3 = Phase3Integration()
        
        for ticker in tickers:
            print(f"\n   Analyzing {ticker}...")
            analysis = phase3.run_phase3_analysis(ticker)
            
            results[ticker] = {
                'risk_score': analysis.combined_risk_score,
                'market_impact': analysis.market_impact_score,
                'confidence': analysis.confidence_level,
                'insights_count': len(analysis.key_insights),
                'recommendations_count': len(analysis.recommendations)
            }
            
            print(f"      Risk: {analysis.combined_risk_score:.1f}/100")
            print(f"      Impact: {analysis.market_impact_score:.1f}/100")
            print(f"      Confidence: {analysis.confidence_level:.1f}/100")
        
        # Compare results
        print(f"\n   📊 Comparison Summary:")
        for ticker, data in results.items():
            risk_level = "Low" if data['risk_score'] < 30 else "Medium" if data['risk_score'] < 60 else "High"
            print(f"      {ticker}: {risk_level} Risk ({data['risk_score']:.1f}/100)")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error in comprehensive analysis: {e}")
        return False

def main():
    """Main test function."""
    print("🚀 Phase 3 Integration Test Suite")
    print("=" * 60)
    print(f"📅 Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Run tests
    tests = [
        ("Individual Services", test_phase3_services),
        ("Integration", test_phase3_integration),
        ("Data Saving", test_data_saving),
        ("Comprehensive Analysis", test_comprehensive_analysis)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            success = test_func()
            results[test_name] = success
            status = "✅ PASSED" if success else "❌ FAILED"
            print(f"\n{status}: {test_name}")
        except Exception as e:
            print(f"\n❌ ERROR: {test_name} - {e}")
            results[test_name] = False
    
    # Summary
    print(f"\n{'='*60}")
    print("📋 TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for success in results.values() if success)
    total = len(results)
    
    for test_name, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status}: {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All Phase 3 tests passed successfully!")
        return 0
    else:
        print("⚠️ Some tests failed. Please check the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
