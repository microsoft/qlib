import Polar from './Polar.js';
export default function polarPrepareCustom(coordSys: Polar): {
    coordSys: {
        type: string;
        cx: number;
        cy: number;
        r: number;
        r0: number;
    };
    api: {
        coord: (data: number[]) => number[];
        size: (dataSize: number[], dataItem: number[]) => number[];
    };
};
