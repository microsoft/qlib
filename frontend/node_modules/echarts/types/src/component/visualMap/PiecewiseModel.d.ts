import VisualMapModel, { VisualMapOption, VisualMeta } from './VisualMapModel.js';
import { VisualMappingOption } from '../../visual/VisualMapping.js';
import { VisualOptionPiecewise } from '../../util/types.js';
import { Dictionary } from 'zrender/lib/core/types.js';
interface VisualPiece extends VisualOptionPiecewise {
    min?: number;
    max?: number;
    lt?: number;
    gt?: number;
    lte?: number;
    gte?: number;
    value?: number;
    label?: string;
}
declare type VisualState = VisualMapModel['stateList'][number];
declare type InnerVisualPiece = VisualMappingOption['pieceList'][number];
/**
 * Order Rule:
 *
 * option.categories / option.pieces / option.text / option.selected:
 *     If !option.inverse,
 *     Order when vertical: ['top', ..., 'bottom'].
 *     Order when horizontal: ['left', ..., 'right'].
 *     If option.inverse, the meaning of
 *     the order should be reversed.
 *
 * this._pieceList:
 *     The order is always [low, ..., high].
 *
 * Mapping from location to low-high:
 *     If !option.inverse
 *     When vertical, top is high.
 *     When horizontal, right is high.
 *     If option.inverse, reverse.
 */
export interface PiecewiseVisualMapOption extends VisualMapOption {
    align?: 'auto' | 'left' | 'right';
    minOpen?: boolean;
    maxOpen?: boolean;
    /**
     * When put the controller vertically, it is the length of
     * horizontal side of each item. Otherwise, vertical side.
     * When put the controller vertically, it is the length of
     * vertical side of each item. Otherwise, horizontal side.
     */
    itemWidth?: number;
    itemHeight?: number;
    itemSymbol?: string;
    pieces?: VisualPiece[];
    /**
     * category names, like: ['some1', 'some2', 'some3'].
     * Attr min/max are ignored when categories set. See "Order Rule"
     */
    categories?: string[];
    /**
     * If set to 5, auto split five pieces equally.
     * If set to 0 and component type not set, component type will be
     * determined as "continuous". (It is less reasonable but for ec2
     * compatibility, see echarts/component/visualMap/typeDefaulter)
     */
    splitNumber?: number;
    /**
     * Object. If not specified, means selected. When pieces and splitNumber: {'0': true, '5': true}
     * When categories: {'cate1': false, 'cate3': true} When selected === false, means all unselected.
     */
    selected?: Dictionary<boolean>;
    selectedMode?: 'multiple' | 'single' | boolean;
    /**
     * By default, when text is used, label will hide (the logic
     * is remained for compatibility reason)
     */
    showLabel?: boolean;
    itemGap?: number;
    hoverLink?: boolean;
}
declare class PiecewiseModel extends VisualMapModel<PiecewiseVisualMapOption> {
    static type: "visualMap.piecewise";
    type: "visualMap.piecewise";
    /**
     * The order is always [low, ..., high].
     * [{text: string, interval: Array.<number>}, ...]
     */
    private _pieceList;
    private _mode;
    optionUpdated(newOption: PiecewiseVisualMapOption, isInit?: boolean): void;
    /**
     * @protected
     * @override
     */
    completeVisualOption(): void;
    private _resetSelected;
    /**
     * @public
     */
    getItemSymbol(): string;
    /**
     * @public
     */
    getSelectedMapKey(piece: InnerVisualPiece): string;
    /**
     * @public
     */
    getPieceList(): InnerVisualPiece[];
    /**
     * @return {string}
     */
    private _determineMode;
    /**
     * @override
     */
    setSelected(selected: this['option']['selected']): void;
    /**
     * @override
     */
    getValueState(value: number): VisualState;
    /**
     * @public
     * @param pieceIndex piece index in visualMapModel.getPieceList()
     */
    findTargetDataIndices(pieceIndex: number): {
        seriesId: string;
        dataIndex: number[];
    }[];
    /**
     * @private
     * @param piece piece.value or piece.interval is required.
     * @return  Can be Infinity or -Infinity
     */
    getRepresentValue(piece: InnerVisualPiece): string | number;
    getVisualMeta(getColorVisual: (value: number, valueState: VisualState) => string): VisualMeta;
    static defaultOption: PiecewiseVisualMapOption;
}
export default PiecewiseModel;
