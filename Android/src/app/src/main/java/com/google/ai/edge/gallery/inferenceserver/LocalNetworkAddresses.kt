package com.google.ai.edge.gallery.inferenceserver

import java.net.Inet4Address
import java.net.NetworkInterface
import java.util.Collections

/** IPv4 addresses assigned to up, non-loopback interfaces (typical Wi‑Fi / LAN). */
object LocalNetworkAddresses {

  fun ipv4Addresses(): List<String> {
    val out = linkedSetOf<String>()
    try {
      for (intf in Collections.list(NetworkInterface.getNetworkInterfaces())) {
        if (!intf.isUp || intf.isLoopback) continue
        for (addr in Collections.list(intf.inetAddresses)) {
          if (addr is Inet4Address && !addr.isLoopbackAddress) {
            addr.hostAddress?.let { out.add(it) }
          }
        }
      }
    } catch (_: Exception) {
    }
    return out.sorted()
  }

  fun httpBaseUrls(port: Int): List<String> = ipv4Addresses().map { "http://$it:$port" }
}
