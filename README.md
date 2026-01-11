# Vibe IK Assistant

Beginner-friendly inverse-kinematics helper that parses natural language, loads robot/tool resources, and solves IK with `roboticstoolbox-python`.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### API key (optional)

Set `OPENAI_API_KEY` in your shell or `.env` file. The key is never committed.

```bash
export OPENAI_API_KEY="your-key"
# or
printf "OPENAI_API_KEY=your-key\n" > .env
```

If the key is missing, the fallback parser uses regex/heuristics and supports the example prompt.

## CLI

```bash
python -m vibeik.cli "I am using the KUKA KR120 R2500 robot. I want to move the tooltip of an 8mm drilling tool to [1.5m, 0.1m, 1.0m]"
```

To get JSON output:

```bash
python -m vibeik.cli "..." --json
```

## API

Start the API:

```bash
uvicorn vibeik.api:app --reload
```

Example request:

```bash
curl -X POST http://127.0.0.1:8000/solve \
  -H 'Content-Type: application/json' \
  -d '{"text":"I am using the KUKA KR120 R2500 robot. I want to move the tooltip of an 8mm drilling tool to [1.5m, 0.1m, 1.0m]"}'
```

Example response:

```json
{
  "ok": true,
  "joint_angles_rad": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
  "joint_angles_deg": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
  "warnings": [],
  "meta": {
    "robot_model": "KR120R2500",
    "tool_name": "Drill_8mm",
    "residual_error": 0.0,
    "solver": "ikine_LM"
  }
}
```

## Resources

Robot and tool definitions live in `RobotResources/`:

- `RobotResources/RobotModels/*.m` must define a DH table named `DH` (6x4).
- `RobotResources/Tools/*.m` must define a tool TCP transform named `T_TCP` (4x4).

Drop new `.m` files into those folders to extend the system.

## Tests

```bash
pytest
```
