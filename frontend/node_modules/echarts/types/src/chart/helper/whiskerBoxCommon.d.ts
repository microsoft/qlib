import type { SeriesOption, SeriesOnCartesianOptionMixin, LayoutOrient } from '../../util/types.js';
import type GlobalModel from '../../model/Global.js';
import type SeriesModel from '../../model/Series.js';
import type SeriesData from '../../data/SeriesData.js';
import type Axis2D from '../../coord/cartesian/Axis2D.js';
import { CoordDimensionDefinition } from '../../data/helper/createDimensions.js';
interface CommonOption extends SeriesOption, SeriesOnCartesianOptionMixin {
    layout?: LayoutOrient;
}
interface WhiskerBoxCommonMixin<Opts extends CommonOption> extends SeriesModel<Opts> {
}
declare class WhiskerBoxCommonMixin<Opts extends CommonOption> {
    /**
     * @private
     * @type {string}
     */
    _baseAxisDim: string;
    defaultValueDimensions: CoordDimensionDefinition['dimsDef'];
    /**
     * @private
     */
    _hasEncodeRule(key: string): boolean;
    /**
     * @override
     */
    getInitialData(option: Opts, ecModel: GlobalModel): SeriesData;
    /**
     * If horizontal, base axis is x, otherwise y.
     * @override
     */
    getBaseAxis(): Axis2D;
}
export { WhiskerBoxCommonMixin };
