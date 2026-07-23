import { expect, expectTypeOf, test } from "vitest";
import * as z from "zod/v4";

test("type inference", () => {
  const booleanRecord = z.record(z.string(), z.boolean());
  type booleanRecord = typeof booleanRecord._output;

  const recordWithEnumKeys = z.record(z.enum(["Tuna", "Salmon"]), z.string());
  type recordWithEnumKeys = z.infer<typeof recordWithEnumKeys>;

  const recordWithLiteralKey = z.record(z.literal(["Tuna", "Salmon", 21]), z.string());
  type recordWithLiteralKey = z.infer<typeof recordWithLiteralKey>;

  const recordWithLiteralUnionKeys = z.record(
    z.union([z.literal("Tuna"), z.literal("Salmon"), z.literal(21)]),
    z.string()
  );
  type recordWithLiteralUnionKeys = z.infer<typeof recordWithLiteralUnionKeys>;

  enum Enum {
    Tuna = 0,
    Salmon = "Shark",
  }

  const recordWithTypescriptEnum = z.record(z.enum(Enum), z.string());
  type recordWithTypescriptEnum = z.infer<typeof recordWithTypescriptEnum>;

  expectTypeOf<booleanRecord>().toEqualTypeOf<Record<string, boolean>>();
  expectTypeOf<recordWithEnumKeys>().toEqualTypeOf<Record<"Tuna" | "Salmon", string>>();
  expectTypeOf<recordWithLiteralKey>().toEqualTypeOf<Record<"Tuna" | "Salmon" | 21, string>>();
  expectTypeOf<recordWithLiteralUnionKeys>().toEqualTypeOf<Record<"Tuna" | "Salmon" | 21, string>>();
  expectTypeOf<recordWithTypescriptEnum>().toEqualTypeOf<Record<Enum, string>>();
});

test("enum exhaustiveness", () => {
  const schema = z.record(z.enum(["Tuna", "Salmon"]), z.string());
  expect(
    schema.parse({
      Tuna: "asdf",
      Salmon: "asdf",
    })
  ).toEqual({
    Tuna: "asdf",
    Salmon: "asdf",
  });

  expect(schema.safeParse({ Tuna: "asdf", Salmon: "asdf", Trout: "asdf" })).toMatchInlineSnapshot(`
    {
      "error": [ZodError: [
      {
        "code": "unrecognized_keys",
        "keys": [
          "Trout"
        ],
        "path": [],
        "message": "Unrecognized key: \\"Trout\\""
      }
    ]],
      "success": false,
    }
  `);
  expect(schema.safeParse({ Tuna: "asdf" })).toMatchInlineSnapshot(`
    {
      "error": [ZodError: [
      {
        "expected": "string",
        "code": "invalid_type",
        "path": [
          "Salmon"
        ],
        "message": "Invalid input: expected string, received undefined"
      }
    ]],
      "success": false,
    }
  `);
});

test("typescript enum exhaustiveness", () => {
  enum BigFish {
    Tuna = 0,
    Salmon = "Shark",
  }

  const schema = z.record(z.enum(BigFish), z.string());
  const value = {
    [BigFish.Tuna]: "asdf",
    [BigFish.Salmon]: "asdf",
  };

  expect(schema.parse(value)).toEqual(value);

  expect(schema.safeParse({ [BigFish.Tuna]: "asdf", [BigFish.Salmon]: "asdf", Trout: "asdf" })).toMatchInlineSnapshot(`
    {
      "error": [ZodError: [
      {
        "code": "unrecognized_keys",
        "keys": [
          "Trout"
        ],
        "path": [],
        "message": "Unrecognized key: \\"Trout\\""
      }
    ]],
      "success": false,
    }
  `);
  expect(schema.safeParse({ [BigFish.Tuna]: "asdf" })).toMatchInlineSnapshot(`
    {
      "error": [ZodError: [
      {
        "expected": "string",
        "code": "invalid_type",
        "path": [
          "Shark"
        ],
        "message": "Invalid input: expected string, received undefined"
      }
    ]],
      "success": false,
    }
  `);
  expect(schema.safeParse({ [BigFish.Salmon]: "asdf" })).toMatchInlineSnapshot(`
    {
      "error": [ZodError: [
      {
        "expected": "string",
        "code": "invalid_type",
        "path": [
          0
        ],
        "message": "Invalid input: expected string, received undefined"
      }
    ]],
      "success": false,
    }
  `);
});

test("literal exhaustiveness", () => {
  const schema = z.record(z.literal(["Tuna", "Salmon", 21]), z.string());
  schema.parse({
    Tuna: "asdf",
    Salmon: "asdf",
    21: "asdf",
  });

  expect(schema.safeParse({ Tuna: "asdf", Salmon: "asdf", 21: "asdf", Trout: "asdf" })).toMatchInlineSnapshot(`
    {
      "error": [ZodError: [
      {
        "code": "unrecognized_keys",
        "keys": [
          "Trout"
        ],
        "path": [],
        "message": "Unrecognized key: \\"Trout\\""
      }
    ]],
      "success": false,
    }
  `);
  expect(schema.safeParse({ Tuna: "asdf" })).toMatchInlineSnapshot(`
    {
      "error": [ZodError: [
      {
        "expected": "string",
        "code": "invalid_type",
        "path": [
          "Salmon"
        ],
        "message": "Invalid input: expected string, received undefined"
      },
      {
        "expected": "string",
        "code": "invalid_type",
        "path": [
          21
        ],
        "message": "Invalid input: expected string, received undefined"
      }
    ]],
      "success": false,
    }
  `);
});

test("pipe exhaustiveness", () => {
  const schema = z.record(z.enum(["Tuna", "Salmon"]).pipe(z.any()), z.string());
  expect(schema.parse({ Tuna: "asdf", Salmon: "asdf" })).toEqual({
    Tuna: "asdf",
    Salmon: "asdf",
  });

  expect(schema.safeParse({ Tuna: "asdf", Salmon: "asdf", Trout: "asdf" })).toMatchInlineSnapshot(`
    {
      "error": [ZodError: [
      {
        "code": "unrecognized_keys",
        "keys": [
          "Trout"
        ],
        "path": [],
        "message": "Unrecognized key: \\"Trout\\""
      }
    ]],
      "success": false,
    }
  `);
  expect(schema.safeParse({ Tuna: "asdf" })).toMatchInlineSnapshot(`
    {
      "error": [ZodError: [
      {
        "expected": "string",
        "code": "invalid_type",
        "path": [
          "Salmon"
        ],
        "message": "Invalid input: expected string, received undefined"
      }
    ]],
      "success": false,
    }
  `);
});

test("union exhaustiveness", () => {
  const schema = z.record(z.union([z.literal("Tuna"), z.literal("Salmon"), z.literal(21)]), z.string());
  expect(schema.parse({ Tuna: "asdf", Salmon: "asdf", 21: "asdf" })).toEqual({
    Tuna: "asdf",
    Salmon: "asdf",
    21: "asdf",
  });

  expect(schema.safeParse({ Tuna: "asdf", Salmon: "asdf", 21: "asdf", Trout: "asdf" })).toMatchInlineSnapshot(`
    {
      "error": [ZodError: [
      {
        "code": "unrecognized_keys",
        "keys": [
          "Trout"
        ],
        "path": [],
        "message": "Unrecognized key: \\"Trout\\""
      }
    ]],
      "success": false,
    }
  `);
  expect(schema.safeParse({ Tuna: "asdf" })).toMatchInlineSnapshot(`
    {
      "error": [ZodError: [
      {
        "expected": "string",
        "code": "invalid_type",
        "path": [
          "Salmon"
        ],
        "message": "Invalid input: expected string, received undefined"
      },
      {
        "expected": "string",
        "code": "invalid_type",
        "path": [
          21
        ],
        "message": "Invalid input: expected string, received undefined"
      }
    ]],
      "success": false,
    }
  `);
});

test("string record parse - pass", () => {
  const schema = z.record(z.string(), z.boolean());
  schema.parse({
    k1: true,
    k2: false,
    1234: false,
  });

  expect(schema.safeParse({ asdf: 1234 }).success).toEqual(false);
  expect(schema.safeParse("asdf")).toMatchInlineSnapshot(`
    {
      "error": [ZodError: [
      {
        "expected": "record",
        "code": "invalid_type",
        "path": [],
        "message": "Invalid input: expected record, received string"
      }
    ]],
      "success": false,
    }
  `);
});

test("key and value getters", () => {
  const rec = z.record(z.string(), z.number());

  rec.keyType.parse("asdf");
  rec.valueType.parse(1234);
});

test("is not vulnerable to prototype pollution", async () => {
  const rec = z.record(
    z.string(),
    z.object({
      a: z.string(),
    })
  );

  const data = JSON.parse(`
    {
      "__proto__": {
        "a": "evil"
      },
      "b": {
        "a": "good"
      }
    }
  `);

  const obj1 = rec.parse(data);
  expect(obj1.a).toBeUndefined();

  const obj2 = rec.safeParse(data);
  expect(obj2.success).toBe(true);
  if (obj2.success) {
    expect(obj2.data.a).toBeUndefined();
  }

  const obj3 = await rec.parseAsync(data);
  expect(obj3.a).toBeUndefined();

  const obj4 = await rec.safeParseAsync(data);
  expect(obj4.success).toBe(true);
  if (obj4.success) {
    expect(obj4.data.a).toBeUndefined();
  }
});

test("dont remove undefined values", () => {
  const result1 = z.record(z.string(), z.any()).parse({ foo: undefined });

  expect(result1).toEqual({
    foo: undefined,
  });
});

test("allow undefined values", () => {
  const schema = z.record(z.string(), z.undefined());

  expect(
    Object.keys(
      schema.parse({
        _test: undefined,
      })
    )
  ).toEqual(["_test"]);
});

test("async parsing", async () => {
  const schema = z
    .record(
      z.string(),
      z
        .string()
        .optional()
        .refine(async () => true)
    )
    .refine(async () => true);

  const data = {
    foo: "bar",
    baz: "qux",
  };
  const result = await schema.safeParseAsync(data);
  expect(result.data).toEqual(data);
});

test("async parsing", async () => {
  const schema = z
    .record(
      z.string(),
      z
        .string()
        .optional()
        .refine(async () => false)
    )
    .refine(async () => false);

  const data = {
    foo: "bar",
    baz: "qux",
  };
  const result = await schema.safeParseAsync(data);
  expect(result.success).toEqual(false);
  expect(result.error).toMatchInlineSnapshot(`
    [ZodError: [
      {
        "code": "custom",
        "path": [
          "foo"
        ],
        "message": "Invalid input"
      },
      {
        "code": "custom",
        "path": [
          "baz"
        ],
        "message": "Invalid input"
      },
      {
        "code": "custom",
        "path": [],
        "message": "Invalid input"
      }
    ]]
  `);
});

test("partial record", () => {
  const schema = z.partialRecord(z.string(), z.string());
  type schema = z.infer<typeof schema>;
  expectTypeOf<schema>().toEqualTypeOf<Partial<Record<string, string>>>();

  const Keys = z.enum(["id", "name", "email"]); //.or(z.never());
  const Person = z.partialRecord(Keys, z.string());
  expectTypeOf<z.infer<typeof Person>>().toEqualTypeOf<Partial<Record<"id" | "name" | "email", string>>>();

  Person.parse({
    id: "123",
    // name: "John",
    // email: "john@example.com",
  });

  Person.parse({
    // id: "123",
    // name: "John",
    email: "john@example.com",
  });

  expect(Person.def.keyType._zod.def.type).toEqual("enum");
});

test("partialRecord with z.literal([key, ...])", () => {
  const Keys = z.literal(["id", "name", "email"]);
  const schema = z.partialRecord(Keys, z.string());
  type Schema = z.infer<typeof schema>;
  expectTypeOf<Schema>().toEqualTypeOf<Partial<Record<"id" | "name" | "email", string>>>();

  // Should parse valid partials
  expect(schema.parse({})).toEqual({});
  expect(schema.parse({ id: "1" })).toEqual({ id: "1" });
  expect(schema.parse({ name: "n", email: "e@example.com" })).toEqual({ name: "n", email: "e@example.com" });

  // Should fail with unrecognized key, error checked via inline snapshot
  expect(schema.safeParse({ foo: "bar" })).toMatchInlineSnapshot(`
    {
      "error": [ZodError: [
      {
        "code": "invalid_key",
        "origin": "record",
        "issues": [
          {
            "code": "invalid_value",
            "values": [
              "id",
              "name",
              "email"
            ],
            "path": [],
            "message": "Invalid option: expected one of \\"id\\"|\\"name\\"|\\"email\\""
          }
        ],
        "path": [
          "foo"
        ],
        "message": "Invalid key in record"
      }
    ]],
      "success": false,
    }
  `);
});
