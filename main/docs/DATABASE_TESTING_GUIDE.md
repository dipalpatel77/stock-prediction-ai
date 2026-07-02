# Database Testing Guide

## 🎯 **How to Test the Database Outside of Main Code**

This guide provides multiple ways to test and verify the database functionality independently of the main application.

## 📋 **Available Testing Tools**

### 1. **Standalone Database Test Suite**

```bash
python test/test_database_standalone.py
```

**What it tests:**

- ✅ Database connection
- ✅ Table creation
- ✅ Data insertion performance (row-by-row vs batch)
- ✅ Pandas optimization
- ✅ Data retrieval
- ✅ Database cleanup
- ✅ Database manager integration

### 2. **Simple SQLite Test**

```bash
python test/test_database_sqlite.py
```

**What it tests:**

- ✅ Direct SQLite functionality
- ✅ Performance testing
- ✅ Main database file verification
- ✅ Sample data operations

### 3. **Database Browser**

```bash
python test/database_browser.py
```

**What it shows:**

- ✅ Database file locations
- ✅ Table structure
- ✅ Record counts
- ✅ Sample data
- ✅ Recent data queries

## 🔍 **Manual Database Testing**

### **Option 1: Using SQLite Command Line**

```bash
# Connect to database
sqlite3 data/stock_data.db

# List tables
.tables

# Describe table structure
.schema angel_one_data

# Count records
SELECT COUNT(*) FROM angel_one_data;

# Show sample data
SELECT * FROM angel_one_data LIMIT 5;

# Show recent data
SELECT ticker, date, close_price FROM angel_one_data
ORDER BY date DESC LIMIT 10;

# Exit
.quit
```

### **Option 2: Using Python Interactive Shell**

```python
import sqlite3
import pandas as pd

# Connect to database
conn = sqlite3.connect('data/stock_data.db')

# List tables
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = cursor.fetchall()
print("Tables:", [table[0] for table in tables])

# Check angel_one_data table
cursor.execute("SELECT COUNT(*) FROM angel_one_data")
count = cursor.fetchone()[0]
print(f"Records in angel_one_data: {count}")

# Show sample data
df = pd.read_sql_query("SELECT * FROM angel_one_data LIMIT 5", conn)
print(df)

# Close connection
conn.close()
```

### **Option 3: Using Database Browser Tools**

- **DB Browser for SQLite**: Download from https://sqlitebrowser.org/
- **SQLiteStudio**: Download from https://sqlitestudio.pl/
- **VS Code SQLite Extension**: Install SQLite extension in VS Code

## 📊 **Database Performance Testing**

### **Test 1: Insertion Performance**

```python
import sqlite3
import time

def test_insertion_performance():
    conn = sqlite3.connect('test_performance.db')
    cursor = conn.cursor()

    # Create table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS test_data (
            id INTEGER PRIMARY KEY,
            ticker TEXT,
            date TEXT,
            price REAL
        )
    """)

    # Test row-by-row insertion
    start_time = time.time()
    for i in range(1000):
        cursor.execute("INSERT INTO test_data (ticker, date, price) VALUES (?, ?, ?)",
                      (f'TEST{i}', '2024-01-01', 100.0 + i))
    conn.commit()
    row_time = time.time() - start_time

    # Clear table
    cursor.execute("DELETE FROM test_data")

    # Test batch insertion
    start_time = time.time()
    data = [(f'TEST{i}', '2024-01-01', 100.0 + i) for i in range(1000)]
    cursor.executemany("INSERT INTO test_data (ticker, date, price) VALUES (?, ?, ?)", data)
    conn.commit()
    batch_time = time.time() - start_time

    print(f"Row-by-row: {row_time:.2f}s")
    print(f"Batch: {batch_time:.2f}s")
    print(f"Improvement: {row_time/batch_time:.1f}x faster")

    conn.close()
```

### **Test 2: Query Performance**

```python
def test_query_performance():
    conn = sqlite3.connect('data/stock_data.db')

    # Test simple query
    start_time = time.time()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM angel_one_data")
    count = cursor.fetchone()[0]
    simple_time = time.time() - start_time

    # Test complex query
    start_time = time.time()
    cursor.execute("""
        SELECT ticker, AVG(close_price), COUNT(*)
        FROM angel_one_data
        GROUP BY ticker
    """)
    results = cursor.fetchall()
    complex_time = time.time() - start_time

    print(f"Simple query: {simple_time:.4f}s")
    print(f"Complex query: {complex_time:.4f}s")
    print(f"Records: {count}")

    conn.close()
```

## 🗂️ **Database File Locations**

The system creates database files in these locations:

- `data/stock_data.db` - Main database
- `test_stock_data.db` - Test database
- `main/data/stock_data.db` - Alternative location

## 📈 **Expected Performance Results**

### **Before Optimization:**

- 235 records: 30+ minutes
- Row-by-row insertion: Very slow
- No batch operations

### **After Optimization:**

- 235 records: 1-3 seconds
- Batch insertion: 10-50x faster
- Pandas to_sql: 100-1000x faster

## 🔧 **Troubleshooting Common Issues**

### **Issue 1: Database File Not Found**

```bash
# Check if database files exist
ls -la data/stock_data.db
ls -la test_stock_data.db

# Create directory if needed
mkdir -p data
```

### **Issue 2: Permission Errors**

```bash
# Check file permissions
ls -la data/stock_data.db

# Fix permissions if needed
chmod 664 data/stock_data.db
```

### **Issue 3: Database Locked**

```bash
# Check for locked database
lsof data/stock_data.db

# Kill processes if needed
kill -9 <process_id>
```

## 📊 **Database Schema Verification**

### **Check Table Structure:**

```sql
PRAGMA table_info(angel_one_data);
```

### **Check Indexes:**

```sql
PRAGMA index_list(angel_one_data);
```

### **Check Database Integrity:**

```sql
PRAGMA integrity_check;
```

## 🎯 **Quick Verification Commands**

### **1. Check Database Status:**

```bash
python test/database_browser.py
```

### **2. Run Performance Tests:**

```bash
python test/test_database_sqlite.py
```

### **3. Full Test Suite:**

```bash
python test/test_database_standalone.py
```

## 📝 **Test Results Interpretation**

### **✅ Good Results:**

- Database connection successful
- Tables created properly
- Data insertion fast (< 5 seconds for 100 records)
- Data retrieval working
- No errors in logs

### **❌ Issues to Watch:**

- Connection failures
- Slow insertion (> 10 seconds for 100 records)
- Missing tables
- Data corruption
- Permission errors

## 🚀 **Performance Benchmarks**

| Operation           | Records | Expected Time | Status |
| ------------------- | ------- | ------------- | ------ |
| Connection          | -       | < 1s          | ✅     |
| Table Creation      | -       | < 1s          | ✅     |
| Insert 100 records  | 100     | < 5s          | ✅     |
| Insert 1000 records | 1000    | < 30s         | ✅     |
| Query 100 records   | 100     | < 1s          | ✅     |
| Complex Query       | 1000    | < 5s          | ✅     |

## 🎉 **Success Criteria**

Your database is working correctly if:

1. ✅ All test scripts run without errors
2. ✅ Database files are created and accessible
3. ✅ Tables have proper structure
4. ✅ Data insertion is fast (< 5 seconds for 100 records)
5. ✅ Data retrieval works correctly
6. ✅ No performance bottlenecks

## 📞 **Getting Help**

If you encounter issues:

1. Run the database browser: `python test/database_browser.py`
2. Check the test results for specific errors
3. Verify database file permissions
4. Check for locked database files
5. Review the performance benchmarks

The database testing tools provide comprehensive verification of all database functionality outside of the main application! 🚀
