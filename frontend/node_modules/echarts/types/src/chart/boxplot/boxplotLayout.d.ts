import type GlobalModel from '../../model/Global.js';
export interface BoxplotItemLayout {
    ends: number[][];
    initBaseline: number;
}
export default function boxplotLayout(ecModel: GlobalModel): void;
