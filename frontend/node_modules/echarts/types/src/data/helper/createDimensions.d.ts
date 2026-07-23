import { DimensionDefinitionLoose, OptionEncode, OptionEncodeValue, EncodeDefaulter, OptionSourceData, DimensionName, DimensionDefinition, DataVisualDimensions, DimensionIndex } from '../../util/types.js';
import SeriesDimensionDefine from '../SeriesDimensionDefine.js';
import { HashMap } from 'zrender/lib/core/util.js';
import OrdinalMeta from '../OrdinalMeta.js';
import { Source } from '../Source.js';
import { SeriesDataSchema } from './SeriesDataSchema.js';
export interface CoordDimensionDefinition extends DimensionDefinition {
    dimsDef?: (DimensionName | {
        name: DimensionName;
        defaultTooltip?: boolean;
    })[];
    otherDims?: DataVisualDimensions;
    ordinalMeta?: OrdinalMeta;
    coordDim?: DimensionName;
    coordDimIndex?: DimensionIndex;
}
export declare type CoordDimensionDefinitionLoose = CoordDimensionDefinition['name'] | CoordDimensionDefinition;
export declare type PrepareSeriesDataSchemaParams = {
    coordDimensions?: CoordDimensionDefinitionLoose[];
    /**
     * Will use `source.dimensionsDefine` if not given.
     */
    dimensionsDefine?: DimensionDefinitionLoose[];
    /**
     * Will use `source.encodeDefine` if not given.
     */
    encodeDefine?: HashMap<OptionEncodeValue, DimensionName> | OptionEncode;
    dimensionsCount?: number;
    /**
     * Make default encode if user not specified.
     */
    encodeDefaulter?: EncodeDefaulter;
    generateCoord?: string;
    generateCoordCount?: number;
    /**
     * If be able to omit unused dimension
     * Used to improve the performance on high dimension data.
     */
    canOmitUnusedDimensions?: boolean;
};
/**
 * For outside usage compat (like echarts-gl are using it).
 */
export declare function createDimensions(source: Source | OptionSourceData, opt?: PrepareSeriesDataSchemaParams): SeriesDimensionDefine[];
/**
 * This method builds the relationship between:
 * + "what the coord sys or series requires (see `coordDimensions`)",
 * + "what the user defines (in `encode` and `dimensions`, see `opt.dimensionsDefine` and `opt.encodeDefine`)"
 * + "what the data source provids (see `source`)".
 *
 * Some guess strategy will be adapted if user does not define something.
 * If no 'value' dimension specified, the first no-named dimension will be
 * named as 'value'.
 *
 * @return The results are always sorted by `storeDimIndex` asc.
 */
export default function prepareSeriesDataSchema(source: Source | OptionSourceData, opt?: PrepareSeriesDataSchemaParams): SeriesDataSchema;
