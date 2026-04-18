# VERS Sign Language Glossary

This document outlines the characteristic hand geometries and gestures recognized by the VERS Sign Language Model (v4.0). The model is trained to recognize both American Sign Language (ASL / WLASL) and Indian Sign Language (ISL) variations for a 15-word emergency vocabulary.

## Core Emergency Signals

These are the primary crisis indicators that trigger `CRITICAL` or `HIGH` threat alerts.

| Intent / Word | System Meaning | ASL/WLASL Geometry | ISL variant Geometry |
| :--- | :--- | :--- | :--- |
| **HELP** | General distress / SOS | All 5 fingers spread wide apart, palm open facing forward. | Both hands clasped together / tight interlocking grip structure. |
| **ACCIDENT** | Incident requiring response | Tight closed fist, all fingers curled inward tightly covering the thumb. | Two closed fists moving toward each other / colliding motion. |
| **MEDICAL** | Need ambulance / doctor | 4 fingers extended and grouped, with thumb tucked sharply across the palm. | Cross shape formed by extending both index fingers. |
| **FIRE** | Fire hazard / smoke | Index finger pointing straight UP, all other fingers curled, thumb sticking out to the side. | Oscillating / flickering motion with fingers spread and waving upwards. |
| **POLICE** | Security threat / crime | Index and middle fingers pointing up (V shape), other fingers curled tight. | Index and middle fingers extended forward. |
| **AMBULANCE** | Extreme medical crisis | Index, middle, and ring fingers extended (3 fingers), thumb and pinky curled. | Similar 3-finger extension with sweeping motion. |
| **DANGER** | Imminent threat | Closed fist with wrist rotated heavily on the z-axis (facing away/down). | Fist crossed over the chest / tightened near the torso. |

## Contextual Signals

These signs provide additional context to an emergency or indicate specific types of injuries.

| Intent / Word | System Meaning | ASL/WLASL Geometry | ISL variant Geometry |
| :--- | :--- | :--- | :--- |
| **PAIN** | Injury / physical distress | Partially clenched claw shape, fingers curved inward toward the palm. | Hand placed flat but arched tightly over the chest/injury site. |
| **FALL** | Slip / trip / collapse | Open hand with all fingers pointing straight DOWN vertically. | Flat palm brushing downward rapidly. |

## Modifiers & Responses

These signs modify an ongoing alert or respond to system prompts.

| Intent / Word | System Meaning | ASL/WLASL Geometry | ISL variant Geometry |
| :--- | :--- | :--- | :--- |
| **STOP** | False alarm / halt | Flat palm facing forward, fingers kept tightly together (not spread). | Similar flat palm but fingers slightly bent forward. |
| **SAFE** | All clear / False alarm | Thumb pointing UP, all other fingers forming a tight fist. | Tight fist with thumb tucked, held steady. |
| **YES** | Affirmative response | Fist with thumb curled over the front (ASL 's'), nodding temporally. | Index finger raised / nodding motion. |
| **NO** | Negative response | Index finger extended horizontally (wagging), rest curled. | Index and middle fingers clipping against the thumb. |
| **PLEASE** | Polite request modifier | Flat palm held against the chest, angled inward with deep z-axis. | Two flat palms pressed together (Namaste-like). |
| **EMERGENCY**| Critical escalation | Pinky and index fingers extended (ILY / horn shape), others curled. | Vigorous waving of the index finger. |

---

### How it works in VERS

The VERS dashboard uses an **Intent Mapper** to logically combine these words into actionable alerts. For example:
- `["HELP", "ACCIDENT"]` ➔ Dispatches an **ACCIDENT [CRITICAL]** alert to emergency contacts.
- `["MEDICAL", "PAIN"]` ➔ Dispatches a **MEDICAL [HIGH]** alert.
- `["SAFE"]` ➔ Automatically downgrades the threat level and registers an all-clear.
