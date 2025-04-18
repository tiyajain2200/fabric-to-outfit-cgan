import tensorflow as tf
from src.models import build_generator, build_discriminator
import os

loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)
LAMBDA = 100

vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
vgg.trainable = False

@tf.function
def perceptual_loss(y_true, y_pred):
    y_true = tf.image.resize(y_true, (224, 224))
    y_pred = tf.image.resize(y_pred, (224, 224))
    return tf.reduce_mean(tf.abs(vgg(y_true) - vgg(y_pred)))

def generator_loss(disc_generated_output, gen_output, target):
    gan_loss = loss_obj(tf.ones_like(disc_generated_output), disc_generated_output)
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
    perceptual = perceptual_loss(target, gen_output)
    total_gen_loss = gan_loss + (LAMBDA * l1_loss) + (10 * perceptual)
    return total_gen_loss

def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = loss_obj(tf.ones_like(disc_real_output)*0.9, disc_real_output)  # Label smoothing
    generated_loss = loss_obj(tf.zeros_like(disc_generated_output), disc_generated_output)
    return real_loss + generated_loss

def train(fabrics, outfits, epochs=100):
    generator = build_generator()
    discriminator = build_discriminator()

    gen_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    disc_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

    dataset = tf.data.Dataset.from_tensor_slices((fabrics, outfits)).shuffle(100).batch(4)

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        for step, (input_image, target_image) in enumerate(dataset):
            with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                gen_output = generator(input_image, training=True)
                disc_real_output = discriminator([input_image, target_image], training=True)
                disc_generated_output = discriminator([input_image, gen_output], training=True)

                gen_loss = generator_loss(disc_generated_output, gen_output, target_image)
                disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

            gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
            gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

            gen_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
            disc_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

            if step % 10 == 0:
                print(f"Step {step}: Gen Loss = {gen_loss:.4f}, Disc Loss = {disc_loss:.4f}")

            checkpoint_dir = "checkpoints"
            os.makedirs(checkpoint_dir, exist_ok=True)
            generator.save(os.path.join(checkpoint_dir, f"generator_epoch_{epoch+1}.h5"))
            discriminator.save(os.path.join(checkpoint_dir, f"discriminator_epoch_{epoch+1}.h5"))
            print(f"✅ Models saved for epoch {epoch+1}")


