import sqlite3
db = sqlite3.connect(r"C:\Users\Skyhi\OneDrive\Desktop\Sm Zero bs\benchmarks\output\gemv_profile.sqlite")
cur = db.cursor()

print("=" * 95)
print("  GPU Kernel Execution Analysis (nsys profile)")
print("=" * 95)

cur.execute("""
SELECT 
    s.value as kernel_name,
    COUNT(*) as instances,
    k.registersPerThread,
    k.gridX, k.blockX,
    ROUND(AVG(k.end - k.start) / 1000.0, 4) as avg_us,
    ROUND(MIN(k.end - k.start) / 1000.0, 4) as min_us,
    ROUND(MAX(k.end - k.start) / 1000.0, 4) as max_us
FROM CUPTI_ACTIVITY_KIND_KERNEL k
LEFT JOIN StringIds s ON k.shortName = s.id
GROUP BY s.value
ORDER BY avg_us DESC
""")

rows = cur.fetchall()
print("\n{:<50} {:>5} {:>5} {:>6} {:>4} {:>10} {:>10} {:>10}".format(
    "Kernel", "#", "Regs", "Grid", "Blk", "Avg(us)", "Min(us)", "Max(us)"))
print("-" * 95)
for row in rows:
    name = (row[0] or "unknown")[:49]
    print("{:<50} {:>5} {:>5} {:>6} {:>4} {:>10.4f} {:>10.4f} {:>10.4f}".format(
        name, row[1], row[2], row[3], row[4], row[5], row[6], row[7]))

# NVTX ranges
print("\n" + "=" * 95)
print("  NVTX Kernel Timing (Excluding Profiler Overhead)")
print("=" * 95)

cur.execute("""
SELECT 
    n.text as range_name,
    COUNT(*) as instances,
    ROUND(AVG(n.end - n.start) / 1000.0, 2) as avg_us,
    ROUND(MIN(n.end - n.start) / 1000.0, 2) as min_us,
    ROUND(MAX(n.end - n.start) / 1000.0, 2) as max_us,
    ROUND(AVG(n.end - n.start) / 1000.0, 2) as median_us
FROM NVTX_EVENTS n
WHERE n.text IS NOT NULL AND n.eventType = 59
GROUP BY n.text
ORDER BY avg_us DESC
""")

for row in cur.fetchall():
    print("\n  Range: {}".format(row[0]))
    print("    Instances: {}".format(row[1]))
    print("    Avg:  {:.2f} us | Min: {:.2f} us | Max: {:.2f} us".format(row[2], row[3], row[4]))

# Per-kernel detail from NVTX + kernel join
print("\n" + "=" * 95)
print("  Detailed: Per-Kernel GPU Execution Time (from NVTX)")
print("=" * 95)

cur.execute("""
SELECT 
    nv.text as range_name,
    COUNT(*) as cnt,
    ROUND(AVG(k.end - k.start) / 1000.0, 2) as gpu_avg_us,
    ROUND(MIN(k.end - k.start) / 1000.0, 2) as gpu_min_us,
    ROUND(MAX(k.end - k.start) / 1000.0, 2) as gpu_max_us
FROM NVTX_EVENTS nv
JOIN CUPTI_ACTIVITY_KIND_KERNEL k ON nv.correlationId = k.correlationId
WHERE nv.text IS NOT NULL AND nv.eventType = 59
GROUP BY nv.text
ORDER BY gpu_avg_us DESC
""")

for row in cur.fetchall():
    print("\n  {}:".format(row[0]))
    print("    Launches: {}".format(row[1]))
    print("    GPU Time - Avg: {:.2f} us | Min: {:.2f} us | Max: {:.2f} us".format(row[2], row[3], row[4]))

db.close()