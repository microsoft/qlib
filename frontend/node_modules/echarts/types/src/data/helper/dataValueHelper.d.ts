import { ParsedValue, DimensionType } from '../../util/types.js';
/**
 * Convert raw the value in to inner value in List.
 *
 * [Performance sensitive]
 *
 * [Caution]: this is the key logic of user value parser.
 * For backward compatibility, do not modify it until you have to!
 */
export declare function parseDataValue(value: any, opt: {
    type?: DimensionType;
}): ParsedValue;
export declare type RawValueParserType = 'number' | 'time' | 'trim';
declare type RawValueParser = (val: unknown) => unknown;
export declare function getRawValueParser(type: RawValueParserType): RawValueParser;
export interface FilterComparator {
    evaluate(val: unknown): boolean;
}
export declare class SortOrderComparator {
    private _incomparable;
    private _resultLT;
    /**
     * @param order by default: 'asc'
     * @param incomparable by default: Always on the tail.
     *        That is, if 'asc' => 'max', if 'desc' => 'min'
     *        See the definition of "incomparable" in [SORT_COMPARISON_RULE].
     */
    constructor(order: 'asc' | 'desc', incomparable: 'min' | 'max');
    evaluate(lval: unknown, rval: unknown): -1 | 0 | 1;
}
declare type OrderRelationOperator = 'lt' | 'lte' | 'gt' | 'gte';
export declare type RelationalOperator = OrderRelationOperator | 'eq' | 'ne';
/**
 * [FILTER_COMPARISON_RULE]
 * `lt`|`lte`|`gt`|`gte`:
 * + rval must be a number. And lval will be converted to number (`numericToNumber`) to compare.
 * `eq`:
 * + If same type, compare with `===`.
 * + If there is one number, convert to number (`numericToNumber`) to compare.
 * + Else return `false`.
 * `ne`:
 * + Not `eq`.
 *
 *
 * [SORT_COMPARISON_RULE]
 * All the values are grouped into three categories:
 * + "numeric" (number and numeric string)
 * + "non-numeric-string" (string that excluding numeric string)
 * + "others"
 * "numeric" vs "numeric": values are ordered by number order.
 * "non-numeric-string" vs "non-numeric-string": values are ordered by ES spec (#sec-abstract-relational-comparison).
 * "others" vs "others": do not change order (always return 0).
 * "numeric" vs "non-numeric-string": "non-numeric-string" is treated as "incomparable".
 * "number" vs "others": "others" is treated as "incomparable".
 * "non-numeric-string" vs "others": "others" is treated as "incomparable".
 * "incomparable" will be seen as -Infinity or Infinity (depends on the settings).
 * MEMO:
 *   Non-numeric string sort makes sense when we need to put the items with the same tag together.
 *   But if we support string sort, we still need to avoid the misleading like `'2' > '12'`,
 *   So we treat "numeric-string" sorted by number order rather than string comparison.
 *
 *
 * [CHECK_LIST_OF_THE_RULE_DESIGN]
 * + Do not support string comparison until required. And also need to
 *   avoid the misleading of "2" > "12".
 * + Should avoid the misleading case:
 *   `" 22 " gte "22"` is `true` but `" 22 " eq "22"` is `false`.
 * + JS bad case should be avoided: null <= 0, [] <= 0, ' ' <= 0, ...
 * + Only "numeric" can be converted to comparable number, otherwise converted to NaN.
 *   See `util/number.ts#numericToNumber`.
 *
 * @return If `op` is not `RelationalOperator`, return null;
 */
export declare function createFilterComparator(op: string, rval?: unknown): FilterComparator;
export {};
