import tensorflow as tf
from src.models import build_generator, build_discriminator
from src.loss import generator_loss, discriminator_loss

def train(dataset, target, epochs=50, batch_size=16):
    generator = build_generator()
    discriminator = build_discriminator()
    
    gen_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    disc_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

    dataset = tf.data.Dataset.from_tensor_slices((dataset, target)).shuffle(500).batch(batch_size)

    for epoch in range(epochs):
        for fabric, outfit in dataset:
            with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                gen_out = generator(fabric, training=True)

                disc_real = discriminator([fabric, outfit], training=True)
                disc_fake = discriminator([fabric, gen_out], training=True)

                gen_loss = generator_loss(disc_fake, gen_out, outfit)
                disc_loss = discriminator_loss(disc_real, disc_fake)

            gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
            gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

            gen_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
            disc_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

        print(f'Epoch {epoch + 1}/{epochs}, Gen Loss: {gen_loss.numpy()}, Disc Loss: {disc_loss.numpy()}')
        generator.save('checkpoints/generator.h5')

    print("Training complete.")
