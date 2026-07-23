import ComponentModel from '../../model/Component.js';
import { AxisModelExtendedInCreator } from '../axisModelCreator.js';
import { AxisModelCommonMixin } from '../axisModelCommonMixin.js';
import Axis2D from './Axis2D.js';
import { AxisBaseOption } from '../axisCommonTypes.js';
import GridModel from './GridModel.js';
import { AxisBaseModel } from '../AxisBaseModel.js';
import { OrdinalSortInfo } from '../../util/types.js';
export declare type CartesianAxisPosition = 'top' | 'bottom' | 'left' | 'right';
export declare type CartesianAxisOption = AxisBaseOption & {
    gridIndex?: number;
    gridId?: string;
    position?: CartesianAxisPosition;
    offset?: number;
    categorySortInfo?: OrdinalSortInfo;
};
export declare type XAXisOption = CartesianAxisOption & {
    mainType?: 'xAxis';
};
export declare type YAXisOption = CartesianAxisOption & {
    mainType?: 'yAxis';
};
export declare class CartesianAxisModel extends ComponentModel<CartesianAxisOption> implements AxisBaseModel<CartesianAxisOption> {
    static type: string;
    axis: Axis2D;
    getCoordSysModel(): GridModel;
}
export interface CartesianAxisModel extends AxisModelCommonMixin<CartesianAxisOption>, AxisModelExtendedInCreator {
}
export default CartesianAxisModel;
