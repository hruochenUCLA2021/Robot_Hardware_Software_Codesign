# NOTE: Collision Rules in MuJoCo MJCF (contype & conaffinity)

This note records the **correct and verified collision-filtering rule** used by **MuJoCo**.
It is written to avoid a very common (and very misleading) mistake.

---

## ✅ The Correct Rule (Authoritative)

Two geoms **A** and **B** will generate contacts **if and only if**:

```
(A.contype & B.conaffinity) ≠ 0
OR
(B.contype & A.conaffinity) ≠ 0
```

👉 **Logical OR, not AND.**  
👉 **One-sided matching is sufficient.**

---

## Meaning of the Fields

### `contype`
- Bitmask describing **which collision groups this geom belongs to**
- Think: *who I can collide AS*

### `conaffinity`
- Bitmask describing **which collision groups this geom accepts collisions from**
- Think: *who I allow to collide WITH me*

---

## Common Wrong Assumption (DO NOT USE)

❌ **Wrong mental model**
```
(A.contype & B.conaffinity) ≠ 0
AND
(B.contype & A.conaffinity) ≠ 0
```

This is **NOT** how MuJoCo works.

---

## Concrete Example

### Floor vs Cylinder

```xml
<geom name="floor" type="plane" contype="1" conaffinity="1"/>
<geom name="cyl"   type="cylinder" contype="2" conaffinity="1"/>
```

Evaluation:

```
floor.contype & cyl.conaffinity = 1 & 1 = 1  → match
cyl.contype   & floor.conaffinity = 2 & 1 = 0 → no match
```

Because MuJoCo uses **OR**, contact **WILL occur**.

---

## How to Guarantee NO Collision

To block collision completely, **both OR terms must be zero**.

```xml
<geom name="cyl"   contype="2" conaffinity="2"/>
<geom name="floor" contype="1" conaffinity="1"/>
```

Result:
```
1 & 2 = 0
2 & 1 = 0
→ No collision
```

---

## Final Takeaway (Memorize This)

> **MuJoCo collision filtering uses OR, not AND.**
>
> **One matching side is enough to generate contact.**
