import MarkerModel, { MarkerOption, MarkerPositionOption } from './MarkerModel.js';
import GlobalModel from '../../model/Global.js';
import { SymbolOptionMixin, ItemStyleOption, SeriesLabelOption, CallbackDataParams, StatesOptionMixin, StatesMixinBase } from '../../util/types.js';
interface MarkPointStateOption {
    itemStyle?: ItemStyleOption;
    label?: SeriesLabelOption;
}
export interface MarkPointDataItemOption extends MarkPointStateOption, StatesOptionMixin<MarkPointStateOption, StatesMixinBase>, SymbolOptionMixin<CallbackDataParams>, MarkerPositionOption {
    name: string;
}
export interface MarkPointOption extends MarkerOption, SymbolOptionMixin<CallbackDataParams>, StatesOptionMixin<MarkPointStateOption, StatesMixinBase>, MarkPointStateOption {
    mainType?: 'markPoint';
    precision?: number;
    data?: MarkPointDataItemOption[];
}
declare class MarkPointModel extends MarkerModel<MarkPointOption> {
    static type: string;
    type: string;
    createMarkerModelFromSeries(markerOpt: MarkPointOption, masterMarkerModel: MarkPointModel, ecModel: GlobalModel): MarkPointModel;
    static defaultOption: MarkPointOption;
}
export default MarkPointModel;
