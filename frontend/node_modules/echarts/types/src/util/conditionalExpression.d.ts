import { OptionDataValue, DimensionLoose, Dictionary } from './types.js';
import { HashMap } from 'zrender/lib/core/util.js';
import { RawValueParserType, RelationalOperator } from '../data/helper/dataValueHelper.js';
/**
 * The structured expression considered:
 * (1) Literal simplicity
 * (2) Semantic displayed clearly
 *
 * Semantic supports:
 * (1) relational expression
 * (2) logical expression
 *
 * For example:
 * ```js
 * {
 *     and: [{
 *         or: [{
 *             dimension: 'Year', gt: 2012, lt: 2019
 *         }, {
 *             dimension: 'Year', '>': 2002, '<=': 2009
 *         }]
 *     }, {
 *         dimension: 'Product', eq: 'Tofu'
 *     }]
 * }
 *
 * { dimension: 'Product', eq: 'Tofu' }
 *
 * {
 *     or: [
 *         { dimension: 'Product', value: 'Tofu' },
 *         { dimension: 'Product', value: 'Biscuit' }
 *     ]
 * }
 *
 * {
 *     and: [true]
 * }
 * ```
 *
 * [PARSER]
 * In an relation expression object, we can specify some built-in parsers:
 * ```js
 * // Trim if string
 * {
 *     parser: 'trim',
 *     eq: 'Flowers'
 * }
 * // Parse as time and enable arithmetic relation comparison.
 * {
 *     parser: 'time',
 *     lt: '2012-12-12'
 * }
 * // Normalize number-like string and make '-' to Null.
 * {
 *     parser: 'time',
 *     lt: '2012-12-12'
 * }
 * // Normalize to number:
 * // + number-like string (like '  123  ') can be converted to a number.
 * // + where null/undefined or other string will be converted to NaN.
 * {
 *     parser: 'number',
 *     eq: 2011
 * }
 * // RegExp, include the feature in SQL: `like '%xxx%'`.
 * {
 *     reg: /^asdf$/
 * }
 * {
 *     reg: '^asdf$' // Serializable reg exp, will be `new RegExp(...)`
 * }
 * ```
 *
 *
 * [EMPTY_RULE]
 * (1) If a relational expression set value as `null`/`undefined` like:
 * `{ dimension: 'Product', lt: undefined }`,
 * The result will be `false` rather than `true`.
 * Consider the case like "filter condition", return all result when null/undefined
 * is probably not expected and even dangours.
 * (2) If a relational expression has no operator like:
 * `{ dimension: 'Product' }`,
 * An error will be thrown. Because it is probably a mistake.
 * (3) If a logical expression has no children like
 * `{ and: undefined }` or `{ and: [] }`,
 * An error will be thrown. Because it is probably an mistake.
 * (4) If intending have a condition that always `true` or always `false`,
 * Use `true` or `flase`.
 * The entire condition can be `true`/`false`,
 * or also can be `{ and: [true] }`, `{ or: [false] }`
 */
/**
 * Date string and ordinal string can be accepted.
 */
interface RelationalExpressionOptionByOp extends Record<RelationalOperator, OptionDataValue> {
    reg?: RegExp | string;
}
declare const RELATIONAL_EXPRESSION_OP_ALIAS_MAP: {
    readonly value: "eq";
    readonly '<': "lt";
    readonly '<=': "lte";
    readonly '>': "gt";
    readonly '>=': "gte";
    readonly '=': "eq";
    readonly '!=': "ne";
    readonly '<>': "ne";
};
declare type RelationalExpressionOptionByOpAlias = Record<keyof typeof RELATIONAL_EXPRESSION_OP_ALIAS_MAP, OptionDataValue>;
interface RelationalExpressionOption extends RelationalExpressionOptionByOp, RelationalExpressionOptionByOpAlias {
    dimension?: DimensionLoose;
    parser?: RawValueParserType;
}
interface LogicalExpressionOption {
    and?: LogicalExpressionSubOption[];
    or?: LogicalExpressionSubOption[];
    not?: LogicalExpressionSubOption;
}
declare type LogicalExpressionSubOption = LogicalExpressionOption | RelationalExpressionOption | TrueFalseExpressionOption;
export declare type TrueExpressionOption = true;
export declare type FalseExpressionOption = false;
export declare type TrueFalseExpressionOption = TrueExpressionOption | FalseExpressionOption;
export declare type ConditionalExpressionOption = LogicalExpressionOption | RelationalExpressionOption | TrueFalseExpressionOption;
declare type ValueGetterParam = Dictionary<unknown>;
export interface ConditionalExpressionValueGetterParamGetter<VGP extends ValueGetterParam = ValueGetterParam> {
    (relExpOption: RelationalExpressionOption): VGP;
}
export interface ConditionalExpressionValueGetter<VGP extends ValueGetterParam = ValueGetterParam> {
    (param: VGP): OptionDataValue;
}
declare class ConditionalExpressionParsed {
    private _cond;
    constructor(exprOption: ConditionalExpressionOption, getters: ConditionalGetters);
    evaluate(): boolean;
}
interface ConditionalGetters<VGP extends ValueGetterParam = ValueGetterParam> {
    prepareGetValue: ConditionalExpressionValueGetterParamGetter<VGP>;
    getValue: ConditionalExpressionValueGetter<VGP>;
    valueGetterAttrMap: HashMap<boolean, string>;
}
export declare function parseConditionalExpression<VGP extends ValueGetterParam = ValueGetterParam>(exprOption: ConditionalExpressionOption, getters: ConditionalGetters<VGP>): ConditionalExpressionParsed;
export {};
