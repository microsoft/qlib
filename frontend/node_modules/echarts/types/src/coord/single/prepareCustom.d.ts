import Single from './Single.js';
export default function singlePrepareCustom(coordSys: Single): {
    coordSys: {
        type: string;
        x: number;
        y: number;
        width: number;
        height: number;
    };
    api: {
        coord: (val: number) => number[];
        size: (dataSize: number | number[], dataItem: number | number[]) => number;
    };
};
