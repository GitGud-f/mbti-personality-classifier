package org.parallelcomp;

import org.parallelcomp.model.Vault;
import org.parallelcomp.model.AscendingHackerThread;
import org.parallelcomp.model.DescendingHackerThread;
import org.parallelcomp.model.RandomHackerThread;
import org.parallelcomp.model.PoliceThread;

import java.util.Random;

public class ConsoleMain {
    public static void consoleVersion() {
        Random rnd = new Random();
        int password = rnd.nextInt(10000);
        Vault vault = new Vault(password);
        System.out.println("Vault password set (for debugging): " + password);

        AscendingHackerThread ascending = new AscendingHackerThread(vault, null);
        DescendingHackerThread descending = new DescendingHackerThread(vault, null);
        RandomHackerThread random = new RandomHackerThread(vault, null);
        PoliceThread police = new PoliceThread(10, null);

        ascending.setPriority(Thread.MAX_PRIORITY);
        descending.setPriority(Thread.MAX_PRIORITY);
        // random.setPriority(Thread.MAX_PRIORITY); // If using

        ascending.start();
        descending.start();
        // random.start(); // If using
        police.start();
    }
}