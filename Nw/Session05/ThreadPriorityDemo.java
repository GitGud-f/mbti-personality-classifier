public class ThreadPriorityDemo {
    
    // Custom thread class
    static class MyThread extends Thread {
        private String threadName;
        
        public MyThread(String name) {
            this.threadName = name;
        }
        
        @Override
        public void run() {
            System.out.println(threadName + " started with priority: " + getPriority());
            
            // Simulate some work
            for (int i = 1; i <= 5; i++) {
                System.out.println(threadName + " - Count: " + i);
                try {
                    Thread.sleep(500); // Sleep for 500ms
                } catch (InterruptedException e) {
                    System.out.println(threadName + " interrupted");
                }
            }
            
            System.out.println(threadName + " finished");
        }
    }
    
    public static void main(String[] args) {
        System.out.println("Main thread started with priority: " + Thread.currentThread().getPriority());
        
        // Create threads with different priorities
        MyThread thread1 = new MyThread("Low Priority Thread");
        MyThread thread2 = new MyThread("Normal Priority Thread");
        MyThread thread3 = new MyThread("High Priority Thread");
        
        // Set thread priorities
        thread1.setPriority(Thread.MIN_PRIORITY);    // Priority 1
        thread2.setPriority(Thread.NORM_PRIORITY);   // Priority 5
        thread3.setPriority(Thread.MAX_PRIORITY);    // Priority 10
        
        // Start the threads
        thread1.start();
        thread2.start();
        thread3.start();
        
        // Wait for all threads to complete
        try {
            thread1.join();
            thread2.join();
            thread3.join();
        } catch (InterruptedException e) {
            System.out.println("Main thread interrupted");
        }
        
        System.out.println("All threads completed!");

    }
}