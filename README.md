# openAI_cartpole_with_q_table
An implementation of table based Q-Learning on OpenAI's CartPole game.

I implemented two different types of q-table. 
The first one uses only the angle and angular velocity of the pole. 
The second one also uses the position of the cart, descretizing into 6 values. 
By doing this, we can expect much higher score.
