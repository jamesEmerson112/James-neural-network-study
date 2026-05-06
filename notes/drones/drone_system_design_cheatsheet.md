# Drone System Design Cheat Sheet

## The 3-Layer Architecture (memorize this)

```
Cloud Server        ←  4G/LTE (seconds latency OK)
  Fleet mgmt, mission assignment, telemetry DB, geofencing

Companion Computer  ←  MAVLink over serial (<10ms)
  Linux + ROS 2: path planning, SLAM, computer vision, obstacle avoidance

Flight Controller   ←  PWM signals to motors
  RTOS (NuttX/ChibiOS): sensor fusion, PID loops at 400Hz, motor mixing
```

Key principle: each layer is a fallback for the one above it.
Cloud dies → Companion takes over. Companion dies → FC keeps flying.

---

## Jargon Quick Reference

| Term | What it is |
|------|-----------|
| PX4 | Open-source firmware for flight controllers (runs on NuttX RTOS) |
| ArduPilot | Alternative open-source FC firmware (runs on ChibiOS RTOS) |
| NuttX | POSIX-flavored RTOS. Like ThreadX but uses `pthread_create()` instead of `tx_thread_create()` |
| MAVLink | Serial message protocol between FC and companion. Structs over UART — heartbeat, GPS, commands |
| MAVSDK | Python/C++ wrapper around MAVLink. `drone.takeoff()` instead of raw bytes |
| ROS 2 | Pub/sub middleware for robots. NOT an OS. Each software module is a "node" |
| Pixhawk | Hardware board the FC firmware runs on (STM32-based) |
| SITL | "Software In The Loop" — full drone simulation on laptop, no hardware |
| GCS | Ground Control Station — laptop/phone app showing drone on a map |
| PID | Proportional-Integral-Derivative controller. Converts error into motor correction at 400Hz |
| IMU | Accelerometer + gyroscope. Tells drone which way is up |
| ESC | Electronic Speed Controller. Sits between FC and each motor |
| Kalman Filter | Math that fuses noisy IMU + GPS + barometer into one reliable position estimate |
| SLAM | Simultaneous Localization and Mapping — "where am I + what's around me" |

---

## Your GT Projects → Real Drone Stack

| Your project | What you built | Real-world equivalent |
|-------------|---------------|----------------------|
| `drone_pid.py` — `pid_thrust()`, `pid_roll()` | PID controller with crosstrack error, integral windup protection | Runs inside Flight Controller on RTOS at 400Hz |
| `drone_pid.py` — `find_parameters_thrust()` (Twiddle) | Auto-tuning PID gains by minimizing error | Factory calibration / ground testing |
| `indiana_drones.py` — `SLAM` class (Omega/Xi matrices) | Graph-based SLAM with landmark tracking | Companion Computer localization module |
| `indiana_drones.py` — `IndianaDronesPlanner.next_move()` | Path planning + tree avoidance (`line_circle_intersect`) | Companion Computer mission planner + obstacle avoidance |
| `warehouse.py` — A* search delivery planning | Grid-based pathfinding with 8-directional movement | Path planning module on companion computer |
| JCM Global — ThreadX RTOS (3 years) | Real-time task scheduling, mutexes, message queues | Same concepts as NuttX/ChibiOS on flight controllers |

---

## Web ↔ Drone Translation Table

| Web concept | Drone equivalent |
|------------|-----------------|
| Frontend (React) | Ground Control Station UI |
| Backend API server | Companion Computer (Linux + ROS 2) |
| Database | SLAM map (Omega/Xi) / Telemetry DB |
| Message queue (Redis pub/sub) | MAVLink messages (serial pub/sub) |
| Microservices | ROS 2 nodes (camera node, planner node, etc.) |
| Load balancer | Sensor fusion (combine noisy inputs into one signal) |
| Health checks / heartbeats | MAVLink heartbeat (FC → "I'm alive" every 1s) |
| Retry logic / circuit breakers | Failsafes (signal lost → return home) |
| Latency SLAs (200ms response) | Real-time deadlines (2.5ms PID loop) |
| CI/CD pipeline | SITL → hardware-in-loop → field test |
| Horizontal scaling | Fleet management |
| API contracts (OpenAPI/REST) | MAVLink message definitions |
| Rate limiting | Motor saturation / integral windup protection |

---

## System Design Answer Template

### 1. Clarify Requirements
- One drone or fleet?
- Max range? Indoor/outdoor?
- What sensors? (camera, LiDAR, GPS)
- Autonomy level? (manual, semi-auto, full auto)
- Regulations? (FAA remote ID, geofencing)

### 2. Draw the Architecture
Three boxes: Cloud ↔ Companion ↔ Flight Controller
Label the communication between each (4G, MAVLink serial, PWM)

### 3. Detail Each Layer

**Flight Controller (RTOS, hard real-time)**
- Sensor fusion: IMU + GPS + barometer → Kalman filter → position estimate
- PID control: position error → motor speeds at 400Hz (2.5ms deadline)
- Failsafes: return-to-home, auto-land on low battery
- Why RTOS: miss a 2.5ms deadline = drone wobbles or crashes

**Companion Computer (Linux, soft real-time)**
- SLAM: build map of environment + track own position
- Path planning: A* or RRT through 3D waypoints
- Computer vision: obstacle detection, landing zone identification
- Mission logic: "am I at the delivery point? release package"
- Sends waypoint commands to FC via MAVLink

**Cloud Server (no real-time requirements)**
- Fleet management: assign missions to drones
- Telemetry database: GPS logs, battery history, flight paths
- Geofencing: no-fly zone enforcement
- Analytics: delivery success rates, maintenance scheduling

### 4. Data Flow (walk through a delivery)
1. Cloud assigns mission → "deliver to (lat, lon)"
2. Companion plans path → A* through 3D waypoints, avoids no-fly zones
3. Companion sends waypoints → MAVLink to FC
4. FC executes → PID loop holds course at 400Hz
5. Camera detects obstacle → Companion replans (~100ms)
6. SLAM updates position → corrects GPS/IMU drift
7. Arrival → FC holds hover via PID
8. Drop package → Companion triggers servo
9. Report delivered → MAVLink → Companion → 4G → Cloud

### 5. Failure Modes (what separates good from great)

| Failure | Response | Who |
|---------|----------|-----|
| GPS lost | Switch to visual odometry (camera) | Companion |
| 4G link lost | Continue mission autonomously | Companion |
| Companion crashes | Return-to-home mode | FC (RTOS) |
| 1 of 4 motors fails | Redistribute thrust, emergency land | FC |
| Low battery | Abort, land at nearest safe point | FC + Companion |
| Obstacle too close | Emergency stop, hover, wait | Companion + FC |

### 6. Tradeoffs to Discuss
- **Edge vs Cloud**: run CV on-drone (low latency, weak GPU) or stream to cloud (high latency, strong GPU)?
- **GPS vs Visual SLAM**: GPS = easy, 3-5m accuracy. Visual SLAM = cm-accurate, compute-heavy
- **Battery vs Payload**: bigger computer = more capability = heavier = less flight time
- **Redundancy vs Cost**: dual IMU, dual GPS = safer but 2x sensor cost
- **Autonomy vs Reliability**: more autonomous = more edge cases to handle

---

## Latency Budgets (key numbers)

| Layer | Deadline | What happens if missed |
|-------|----------|----------------------|
| PID loop (FC) | 2.5ms (400Hz) | Drone wobbles, potentially crashes |
| Sensor fusion (FC) | 5ms (200Hz) | Position estimate drifts |
| Obstacle avoidance (Companion) | 50-100ms | Might clip an obstacle |
| Path replanning (Companion) | 100-500ms | Takes a suboptimal route |
| Cloud telemetry | Seconds | Dashboard is stale, not dangerous |
| Mission assignment (Cloud) | Seconds-minutes | Drone waits, no safety issue |

---

## If They Ask About Specific Subsystems

**"How does the PID controller work?"**
→ You literally built this. Error = current_position - target_position.
Output = -(Kp * error) - (Kd * error_derivative) - (Ki * error_integral).
Integral windup: reset integral term when motor saturates (your `max_rpm_reached` flag).

**"How does SLAM work?"**
→ You built this too. Omega matrix = information (inverse covariance).
Xi vector = weighted position estimates. New measurement → expand matrices,
add constraint. New movement → expand, add motion constraint, factor out old pose.
Solve: position = Omega_inverse * Xi.

**"How do you avoid obstacles?"**
→ Your `line_circle_intersect()` — check if planned path segment intersects
any obstacle circle. If yes, reduce distance and adjust steering angle iteratively
until collision-free path found.

**"Why not just use GPS?"**
→ GPS is 3-5m accuracy, updates at 1-10Hz, and can drop out near buildings.
For a drone that needs to hover within 10cm of a target, you need sensor fusion:
GPS (global but noisy) + IMU (fast but drifts) + barometer (altitude) → Kalman filter
→ reliable position at 200Hz+.
