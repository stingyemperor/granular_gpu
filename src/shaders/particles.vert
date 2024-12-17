#version 120

uniform float pointRadius;  // point size in world space

attribute float pointSize;
varying float fs_pointSize;

varying vec3 fs_PosEye;		// center of the viewpoint space

uniform float pointScale;
varying mat4 u_Persp;

void main(void) {

	vec3 posEye = (gl_ModelViewMatrix  * vec4(gl_Vertex.xyz, 1.0f)).xyz;
	float dist = length(posEye);


	gl_PointSize = pointRadius * pointScale/ dist;

	fs_PosEye = posEye;

	gl_FrontColor = gl_Color;

	u_Persp = gl_ProjectionMatrix;

	gl_Position = ftransform();
	fs_pointSize = pointSize;
}
