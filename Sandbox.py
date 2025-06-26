import time

start_time = time.time()

fib = 0
for i in range (1000000):
    fib += i

end_time = time.time()

print(fib)
print(end_time - start_time)