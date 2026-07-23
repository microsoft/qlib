import { HashMap } from 'zrender/lib/core/util.js';
import { SourceFormat, SeriesLayoutBy, DimensionDefinition, OptionEncodeValue, OptionSourceData, DimensionName, OptionSourceHeader, DimensionDefinitionLoose } from '../util/types.js';
import { DatasetOption } from '../component/dataset/install.js';
/**
 * [sourceFormat]
 *
 * + "original":
 * This format is only used in series.data, where
 * itemStyle can be specified in data item.
 *
 * + "arrayRows":
 * [
 *     ['product', 'score', 'amount'],
 *     ['Matcha Latte', 89.3, 95.8],
 *     ['Milk Tea', 92.1, 89.4],
 *     ['Cheese Cocoa', 94.4, 91.2],
 *     ['Walnut Brownie', 85.4, 76.9]
 * ]
 *
 * + "objectRows":
 * [
 *     {product: 'Matcha Latte', score: 89.3, amount: 95.8},
 *     {product: 'Milk Tea', score: 92.1, amount: 89.4},
 *     {product: 'Cheese Cocoa', score: 94.4, amount: 91.2},
 *     {product: 'Walnut Brownie', score: 85.4, amount: 76.9}
 * ]
 *
 * + "keyedColumns":
 * {
 *     'product': ['Matcha Latte', 'Milk Tea', 'Cheese Cocoa', 'Walnut Brownie'],
 *     'count': [823, 235, 1042, 988],
 *     'score': [95.8, 81.4, 91.2, 76.9]
 * }
 *
 * + "typedArray"
 *
 * + "unknown"
 */
export interface SourceMetaRawOption {
    seriesLayoutBy: SeriesLayoutBy;
    sourceHeader: OptionSourceHeader;
    dimensions: DimensionDefinitionLoose[];
}
export interface Source extends SourceImpl {
}
declare class SourceImpl {
    /**
     * Not null/undefined.
     */
    readonly data: OptionSourceData;
    /**
     * See also "detectSourceFormat".
     * Not null/undefined.
     */
    readonly sourceFormat: SourceFormat;
    /**
     * 'row' or 'column'
     * Not null/undefined.
     */
    readonly seriesLayoutBy: SeriesLayoutBy;
    /**
     * dimensions definition from:
     * (1) standalone defined in option prop `dimensions: [...]`
     * (2) detected from option data. See `determineSourceDimensions`.
     * If can not be detected (e.g., there is only pure data `[[11, 33], ...]`
     * `dimensionsDefine` will be null/undefined.
     */
    readonly dimensionsDefine: DimensionDefinition[];
    /**
     * Only make sense in `SOURCE_FORMAT_ARRAY_ROWS`.
     * That is the same as `sourceHeader: number`,
     * which means from which line the real data start.
     * Not null/undefined, uint.
     */
    readonly startIndex: number;
    /**
     * Dimension count detected from data. Only works when `dimensionDefine`
     * does not exists.
     * Can be null/undefined (when unknown), uint.
     */
    readonly dimensionsDetectedCount: number;
    /**
     * Raw props from user option.
     */
    readonly metaRawOption: SourceMetaRawOption;
    constructor(fields: {
        data: OptionSourceData;
        sourceFormat: SourceFormat;
        seriesLayoutBy?: SeriesLayoutBy;
        dimensionsDefine?: DimensionDefinition[];
        startIndex?: number;
        dimensionsDetectedCount?: number;
        metaRawOption?: SourceMetaRawOption;
        encodeDefine?: HashMap<OptionEncodeValue, DimensionName>;
    });
}
export declare function isSourceInstance(val: unknown): val is Source;
/**
 * Create a source from option.
 * NOTE: Created source is immutable. Don't change any properties in it.
 */
export declare function createSource(sourceData: OptionSourceData, thisMetaRawOption: SourceMetaRawOption, sourceFormat: SourceFormat): Source;
/**
 * Wrap original series data for some compatibility cases.
 */
export declare function createSourceFromSeriesDataOption(data: OptionSourceData): Source;
/**
 * Clone source but excludes source data.
 */
export declare function cloneSourceShallow(source: Source): Source;
/**
 * Note: An empty array will be detected as `SOURCE_FORMAT_ARRAY_ROWS`.
 */
export declare function detectSourceFormat(data: DatasetOption['source']): SourceFormat;
export declare function shouldRetrieveDataByName(source: Source): boolean;
export {};
