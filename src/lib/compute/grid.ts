import nodes from '$lib/node_register';
import categories from '$lib/node_categories';
import { readFileFromGCS } from '$lib/utils';

interface BaseData {}

interface GridData extends BaseData {
	text: string;
	output: object;
}

export default class GridNode {
	id: string;
	data: GridData;
	position: { x: number; y: number };
	type: string;

	constructor(node_data) {
		const { id, data, position, type } = node_data;
		this.id = id;
		this.data = data;
		this.position = position;
		this.type = type;
	}

	async compute(
		inputData: object,
		context: string,
		info: (arg: string) => void,
		error: (arg: string) => void,
		success: (arg: string) => void,
		slug: string
	) {
		if (!this.data.dirty && this.data.output && Object.keys(this.data.output).length > 0) {
			return this.data.output;
		}

		const input = inputData[Object.keys(inputData)[0]];
		this.data.output = input;
		this.data.dirty = false;
		return this.data.output;
	}
}

type GridNodeInterface = DGNodeInterface & {
	data: GridData;
};

export let grid_node_data: GridNodeInterface = {
	id: 'grid',
	data: {
		label: 'Grid Node',
		text: '',
		dirty: false,
		output: {},
		compute_type: 'grid_v0',
		input_ids: {},
		category: categories.display.id,
		icon: 'grid_v0'
	},
	position: { x: 0, y: 0 },
	type: 'grid_v0'
};

export let grid_node = new GridNode(grid_node_data);

nodes.register(GridNode, grid_node_data);
