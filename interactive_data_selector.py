#!/usr/bin/env python3
"""
Interactive Data Selector for Angel One API
Allows users to select data period and training data size
"""

import sys
import os
sys.path.insert(0, '.')

from datetime import datetime, timedelta
import pandas as pd

class InteractiveDataSelector:
    """
    Interactive interface for selecting data parameters
    """
    
    def __init__(self):
        self.angel_one_limits = {
            'ONE_MINUTE': 30,
            'THREE_MINUTE': 60,
            'FIVE_MINUTE': 100,
            'TEN_MINUTE': 100,
            'FIFTEEN_MINUTE': 200,
            'THIRTY_MINUTE': 200,
            'ONE_HOUR': 400,
            'ONE_DAY': 2000
        }
        
        self.interval_descriptions = {
            'ONE_MINUTE': '1 Minute (Intraday)',
            'THREE_MINUTE': '3 Minutes (Intraday)',
            'FIVE_MINUTE': '5 Minutes (Intraday)',
            'TEN_MINUTE': '10 Minutes (Intraday)',
            'FIFTEEN_MINUTE': '15 Minutes (Intraday)',
            'THIRTY_MINUTE': '30 Minutes (Intraday)',
            'ONE_HOUR': '1 Hour (Intraday)',
            'ONE_DAY': '1 Day (Daily)'
        }
    
    def display_welcome(self):
        """Display welcome message and options"""
        print("🚀 INTERACTIVE DATA SELECTOR")
        print("=" * 50)
        print("📊 Angel One API Data Configuration")
        print("🎯 Select your data parameters for optimal analysis")
        print()
    
    def select_interval(self):
        """Let user select data interval"""
        print("📈 SELECT DATA INTERVAL:")
        print("-" * 30)
        
        options = []
        for i, (interval, description) in enumerate(self.interval_descriptions.items(), 1):
            max_days = self.angel_one_limits[interval]
            print(f"   {i}. {description}")
            print(f"      📅 Max days: {max_days:,} days")
            print(f"      🎯 Best for: {'Intraday trading' if 'MINUTE' in interval or 'HOUR' in interval else 'Long-term analysis'}")
            print()
            options.append(interval)
        
        while True:
            try:
                choice = input("🔢 Enter your choice (1-8): ").strip()
                choice_num = int(choice)
                
                if 1 <= choice_num <= 8:
                    selected_interval = options[choice_num - 1]
                    max_days = self.angel_one_limits[selected_interval]
                    print(f"✅ Selected: {self.interval_descriptions[selected_interval]}")
                    print(f"📅 Maximum available: {max_days:,} days")
                    return selected_interval, max_days
                else:
                    print("❌ Please enter a number between 1 and 8")
            except ValueError:
                print("❌ Please enter a valid number")
    
    def select_data_period(self, max_days):
        """Let user select how many days of data to fetch"""
        print(f"\n📅 SELECT DATA PERIOD:")
        print("-" * 25)
        print(f"📊 Maximum available: {max_days:,} days")
        print()
        
        # Suggest common periods
        suggestions = {
            '1': (7, '1 Week (Quick analysis)'),
            '2': (30, '1 Month (Short-term)'),
            '3': (90, '3 Months (Medium-term)'),
            '4': (180, '6 Months (Long-term)'),
            '5': (365, '1 Year (Comprehensive)'),
            '6': (max_days, f'Maximum ({max_days:,} days)')
        }
        
        print("💡 SUGGESTED PERIODS:")
        for key, (days, description) in suggestions.items():
            if days <= max_days:
                print(f"   {key}. {description}")
        print(f"   7. Custom period")
        print()
        
        while True:
            try:
                choice = input("🔢 Enter your choice (1-7): ").strip()
                
                if choice in suggestions:
                    days = suggestions[choice][0]
                    if days <= max_days:
                        print(f"✅ Selected: {suggestions[choice][1]}")
                        return days
                    else:
                        print(f"❌ {suggestions[choice][1]} exceeds maximum of {max_days:,} days")
                elif choice == '7':
                    # Custom period
                    while True:
                        try:
                            custom_days = int(input(f"📅 Enter custom days (1-{max_days:,}): "))
                            if 1 <= custom_days <= max_days:
                                print(f"✅ Selected: {custom_days:,} days")
                                return custom_days
                            else:
                                print(f"❌ Please enter between 1 and {max_days:,} days")
                        except ValueError:
                            print("❌ Please enter a valid number")
                else:
                    print("❌ Please enter a number between 1 and 7")
            except ValueError:
                print("❌ Please enter a valid number")
    
    def select_training_data_size(self, total_days):
        """Let user select how much data to use for training"""
        print(f"\n🎯 SELECT TRAINING DATA SIZE:")
        print("-" * 30)
        print(f"📊 Total data available: {total_days:,} days")
        print()
        
        # Calculate suggested training sizes
        suggestions = {
            '1': (min(30, total_days), '30 days (Quick training)'),
            '2': (min(90, total_days), '90 days (Standard training)'),
            '3': (min(180, total_days), '180 days (Comprehensive training)'),
            '4': (min(365, total_days), '365 days (Full year training)'),
            '5': (total_days, f'All data ({total_days:,} days)')
        }
        
        print("💡 SUGGESTED TRAINING SIZES:")
        for key, (days, description) in suggestions.items():
            if days <= total_days:
                print(f"   {key}. {description}")
        print(f"   6. Custom training size")
        print()
        
        while True:
            try:
                choice = input("🔢 Enter your choice (1-6): ").strip()
                
                if choice in suggestions:
                    training_days = suggestions[choice][0]
                    if training_days <= total_days:
                        remaining_days = total_days - training_days
                        print(f"✅ Selected: {suggestions[choice][1]}")
                        print(f"📊 Training data: {training_days:,} days")
                        print(f"📊 Remaining data: {remaining_days:,} days")
                        return training_days
                    else:
                        print(f"❌ {suggestions[choice][1]} exceeds available data")
                elif choice == '6':
                    # Custom training size
                    while True:
                        try:
                            custom_training = int(input(f"🎯 Enter training days (1-{total_days:,}): "))
                            if 1 <= custom_training <= total_days:
                                remaining_days = total_days - custom_training
                                print(f"✅ Selected: {custom_training:,} days for training")
                                print(f"📊 Training data: {custom_training:,} days")
                                print(f"📊 Remaining data: {remaining_days:,} days")
                                return custom_training
                            else:
                                print(f"❌ Please enter between 1 and {total_days:,} days")
                        except ValueError:
                            print("❌ Please enter a valid number")
                else:
                    print("❌ Please enter a number between 1 and 6")
            except ValueError:
                print("❌ Please enter a valid number")
    
    def select_prediction_horizon(self):
        """Let user select prediction horizon"""
        print(f"\n🔮 SELECT PREDICTION HORIZON:")
        print("-" * 30)
        
        horizons = {
            '1': (1, '1 day (Next day)'),
            '2': (3, '3 days (Short-term)'),
            '3': (7, '1 week (Medium-term)'),
            '4': (14, '2 weeks (Long-term)'),
            '5': (30, '1 month (Extended)')
        }
        
        print("💡 PREDICTION HORIZONS:")
        for key, (days, description) in horizons.items():
            print(f"   {key}. {description}")
        print(f"   6. Custom horizon")
        print()
        
        while True:
            try:
                choice = input("🔢 Enter your choice (1-6): ").strip()
                
                if choice in horizons:
                    horizon_days = horizons[choice][0]
                    print(f"✅ Selected: {horizons[choice][1]}")
                    return horizon_days
                elif choice == '6':
                    # Custom horizon
                    while True:
                        try:
                            custom_horizon = int(input("🔮 Enter prediction days (1-365): "))
                            if 1 <= custom_horizon <= 365:
                                print(f"✅ Selected: {custom_horizon} days prediction horizon")
                                return custom_horizon
                            else:
                                print("❌ Please enter between 1 and 365 days")
                        except ValueError:
                            print("❌ Please enter a valid number")
                else:
                    print("❌ Please enter a number between 1 and 6")
            except ValueError:
                print("❌ Please enter a valid number")
    
    def display_summary(self, interval, data_days, training_days, prediction_days):
        """Display final configuration summary"""
        print(f"\n📋 CONFIGURATION SUMMARY:")
        print("=" * 40)
        print(f"📈 Data Interval: {self.interval_descriptions[interval]}")
        print(f"📅 Data Period: {data_days:,} days")
        print(f"🎯 Training Data: {training_days:,} days")
        print(f"🔮 Prediction Horizon: {prediction_days} days")
        print(f"📊 Remaining Data: {data_days - training_days:,} days")
        print()
        
        # Calculate data efficiency
        training_ratio = (training_days / data_days) * 100
        print(f"📊 Data Efficiency:")
        print(f"   • Training ratio: {training_ratio:.1f}%")
        print(f"   • Validation data: {100 - training_ratio:.1f}%")
        print()
        
        # Performance estimates
        print(f"⚡ Performance Estimates:")
        if interval == 'ONE_DAY':
            print(f"   • Analysis time: ~2-5 minutes")
            print(f"   • Best for: Long-term predictions")
        elif 'MINUTE' in interval:
            print(f"   • Analysis time: ~5-15 minutes")
            print(f"   • Best for: Intraday trading")
        else:
            print(f"   • Analysis time: ~3-8 minutes")
            print(f"   • Best for: Short-term analysis")
        
        print()
        return {
            'interval': interval,
            'data_days': data_days,
            'training_days': training_days,
            'prediction_days': prediction_days,
            'training_ratio': training_ratio
        }
    
    def run_interactive_selection(self):
        """Run the complete interactive selection process"""
        self.display_welcome()
        
        # Step 1: Select interval
        interval, max_days = self.select_interval()
        
        # Step 2: Select data period
        data_days = self.select_data_period(max_days)
        
        # Step 3: Select training data size
        training_days = self.select_training_data_size(data_days)
        
        # Step 4: Select prediction horizon
        prediction_days = self.select_prediction_horizon()
        
        # Step 5: Display summary
        config = self.display_summary(interval, data_days, training_days, prediction_days)
        
        return config

def main():
    """Main function to run interactive selection"""
    try:
        selector = InteractiveDataSelector()
        config = selector.run_interactive_selection()
        
        print("✅ Configuration complete!")
        print("🚀 Ready to run analysis with your selected parameters!")
        
        return config
        
    except KeyboardInterrupt:
        print("\n\n❌ Selection cancelled by user")
        return None
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return None

if __name__ == "__main__":
    config = main()
    if config:
        print(f"\n📋 Final Configuration: {config}")
